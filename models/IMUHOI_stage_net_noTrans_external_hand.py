import torch
from typing import Any, Dict, Optional

from models.IMUHOI_stage_net_noTrans import (
    ObjectTransModule,
    TransPoseNet,
)


class TransPoseNetExternalHand(TransPoseNet):
    """
    Variant of TransPoseNet that skips the internal human pose module and expects
    externally provided hand global positions (e.g. from another human pose model).
    The network still executes the velocity/contact (stage 1) and object translation
    (stage 3) modules to predict HOI/object trajectories.
    """

    def __init__(
        self,
        cfg: Any,
        pretrained_modules: Optional[Dict[str, str]] = None,
        skip_modules: Optional[list] = None,
    ):
        skip = set(skip_modules or [])
        skip.add("human_pose")
        super().__init__(
            cfg,
            pretrained_modules=pretrained_modules,
            skip_modules=list(skip),
        )

    def prepare_inputs(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Wrapper around the base format_input for clarity."""
        return self.format_input(data_dict)

    def _ensure_tensor(
        self,
        value: Any,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        return tensor

    def _prepare_external_hand(
        self,
        hand_glb_pos: Any,
        formatted_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Normalize the externally provided hand global positions to the expected
        shape [B, T, 2, 3] and device/dtype.
        """
        human_imu = formatted_inputs["human_imu"]
        hand_tensor = self._ensure_tensor(hand_glb_pos, human_imu)

        if hand_tensor.dim() == 3:
            # Assume missing batch dimension
            hand_tensor = hand_tensor.unsqueeze(0)

        if hand_tensor.dim() != 4 or hand_tensor.shape[-2:] != (2, 3):
            raise ValueError(
                f"external hand positions must have shape [B, T, 2, 3], "
                f"got {tuple(hand_tensor.shape)}"
            )

        expected_bs, expected_len = human_imu.shape[:2]
        if hand_tensor.shape[0] != expected_bs or hand_tensor.shape[1] != expected_len:
            raise ValueError(
                "Mismatch between external hand positions and formatted inputs: "
                f"expected batch/seq ({expected_bs}, {expected_len}) "
                f"but got ({hand_tensor.shape[0]}, {hand_tensor.shape[1]})"
            )

        return hand_tensor.contiguous()

    def run_velocity_contact(
        self,
        formatted_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the velocity/contact module (stage 1).
        """
        if self.velocity_contact_module is None:
            raise RuntimeError("Velocity/contact module is not initialized or was skipped.")

        return self.velocity_contact_module(
            formatted_inputs["human_imu"],
            formatted_inputs["obj_imu"],
            formatted_inputs["hand_vel_glb_init"],
            formatted_inputs["obj_vel_init"],
            contact_init=formatted_inputs["contact_init"],
        )

    def run_object_trans(
        self,
        formatted_inputs: Dict[str, torch.Tensor],
        pred_hand_glb_pos: torch.Tensor,
        vc_out: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the object translation module (stage 3) using provided hand
        positions and velocity/contact outputs.
        """
        if (not use_object_data) or self.object_trans_module is None:
            batch_size, seq_len = pred_hand_glb_pos.shape[:2]
            device = pred_hand_glb_pos.device
            return ObjectTransModule.empty_output(batch_size, seq_len, device)

        has_object_mask = formatted_inputs["has_object"]
        if not has_object_mask.any():
            batch_size, seq_len = pred_hand_glb_pos.shape[:2]
            device = pred_hand_glb_pos.device
            return ObjectTransModule.empty_output(batch_size, seq_len, device)

        return self.object_trans_module(
            pred_hand_glb_pos,
            vc_out["pred_hand_contact_prob"],
            formatted_inputs["obj_trans_init"],
            obj_imu=formatted_inputs["obj_imu"],
            human_imu=formatted_inputs["human_imu"],
            obj_vel_input=vc_out["pred_obj_vel"],
            contact_init=formatted_inputs["contact_init"],
            has_object_mask=has_object_mask,
            compute_fk=compute_fk,
        )

    def forward(
        self,
        data_dict: Dict[str, Any],
        external_hand_glb_pos: Optional[Any] = None,
        use_object_data: Optional[bool] = None,
        compute_fk: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run stage 1 and stage 3, substituting external hand positions for the
        internal human pose module.
        """
        formatted = self.prepare_inputs(data_dict)
        results: Dict[str, torch.Tensor] = {}

        vc_out = self.run_velocity_contact(formatted)
        results.update(vc_out)

        if external_hand_glb_pos is None:
            raise ValueError("external_hand_glb_pos must be provided for the external-hand pipeline.")
        pred_hand_glb_pos = self._prepare_external_hand(external_hand_glb_pos, formatted)
        results["pred_hand_glb_pos"] = pred_hand_glb_pos

        if use_object_data is None:
            use_object_data = data_dict.get("use_object_data", True)

        obj_out = self.run_object_trans(
            formatted,
            pred_hand_glb_pos,
            vc_out,
            use_object_data=bool(use_object_data),
            compute_fk=compute_fk,
        )
        results.update(obj_out)

        results["has_object"] = formatted["has_object"]
        return results
