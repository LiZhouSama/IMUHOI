import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import inspect
import diffusers
from einops import rearrange # Needed for LearnedSinusoidalPosEmb if used

# ==============================================================================
# Copied Modules from OMOMO Ref Code (transformer_module.py)
# ==============================================================================

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    # Convert to torch.FloatTensor here
    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # seq shape: [bs, seq_len]
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1): # Added dropout arg
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        # Initialize weights (optional but good practice)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(dropout) # Use dropout arg

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout) # Use dropout arg
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        # Reshape and permute: BS x T x D -> BS x T x H x Dh -> H x BS x T x Dh -> (H*BS) x T x Dh
        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(0, 2, 1, 3).contiguous().view(-1, n_q, self.d_k) # Changed permute for batch_first=True consistency later? No, original OMOMO seems batch second for MHA internal
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(0, 2, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(0, 2, 1, 3).contiguous().view(-1, n_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) # (n_head*bs) X n_q X n_k
        attn = attn / self.temperature

        if mask is not None:
            # Expand mask to match attention shape (n_head*bs, n_q, n_k)
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).view(-1, n_q, n_k)
            attn = attn.masked_fill(mask, -float('inf')) # Use float('-inf')

        attn = F.softmax(attn, dim=2) # (n_head*bs) X n_q X n_k
        
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) # (n_head*bs) X n_q X d_v

        # Reshape back: (H*BS) x T x Dv -> H x BS x T x Dv -> BS x T x H x Dv -> BS x T x (H*Dv)
        output = output.view(self.n_head, bs, n_q, self.d_v).permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*D)

        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1): # Added dropout arg
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # Use Conv1d like OMOMO
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout) # Use dropout arg

    def forward(self, x):
        # x: BS X T X D
        residual = x
        output = x.transpose(1, 2) # BS X D X T
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X T
        output = output.transpose(1, 2) # BS X T X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X T X D

        return output

class DecoderLayer(nn.Module):
    """ OMOMO style Decoder Layer """
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1): # Added d_ff, dropout
        super(DecoderLayer, self).__init__()
        # Note: OMOMO DecoderLayer doesn't have cross-attention, only self-attention
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout) # Use d_ff

    def forward(self, dec_input, slf_attn_mask=None): # Simplified inputs based on OMOMO usage
        # decode_input: BS X T X D
        # slf_attn_mask: BS X T X T (masking subsequent positions)
        
        # Apply self-attention
        # OMOMO doesn't seem to use padding mask here, relies on later steps? Or assumes fixed length?
        # We might need padding mask if sequences have variable lengths.
        dec_output, dec_slf_attn = self.self_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)

        # Apply position-wise feedforward
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn

# ==============================================================================
# Timestep Embedding (Standard Implementation)
# ==============================================================================
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        device = t.device
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# ==============================================================================
# Main MotionDiffusion Model using OMOMO Decoder
# ==============================================================================
class IMUEncoder(nn.Module):
    """高级IMU编码器，分别处理加速度和方向数据"""
    def __init__(self, acc_dim=3, ori_dim=6, hidden_dim=64, output_dim=32, dropout=0.1):
        super().__init__()
        
        # 加速度处理分支
        self.acc_encoder = nn.Sequential(
            nn.Linear(acc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 6D旋转表示处理分支
        self.ori_encoder = nn.Sequential(
            nn.Linear(ori_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        参数:
            x: IMU数据 [..., 9]，前3维是加速度，后6维是6D旋转表示
        
        返回:
            编码特征 [..., output_dim]
        """
        # 分离加速度和方向数据
        acc = x[..., :3]
        ori = x[..., 3:]
        
        # 分别编码
        acc_feat = self.acc_encoder(acc)
        ori_feat = self.ori_encoder(ori)
        
        # 拼接并融合
        combined = torch.cat([acc_feat, ori_feat], dim=-1)
        return self.fusion(combined)

class MotionDiffusion(nn.Module):
    def __init__(self, cfg, input_length, imu_input=True):
        super(MotionDiffusion, self).__init__()
        self.cfg = cfg
        self.scheduler = diffusers.DDIMScheduler(**cfg.scheduler.get("params", dict()))
        self.latent_dim = cfg.model.d_model # d_model from config used as latent_dim
        self.imu_input = imu_input
        
        # Transformer parameters from cfg (matching OMOMO structure)
        self.num_layers = cfg.model.num_layers # n_dec_layers in OMOMO
        self.num_heads = cfg.model.n_heads    # n_head in OMOMO
        # d_k and d_v are often derived from d_model / n_head
        self.d_k = cfg.model.get('d_k', self.latent_dim // self.num_heads) 
        self.d_v = cfg.model.get('d_v', self.latent_dim // self.num_heads)
        self.d_ff = cfg.model.get('dim_feedforward', self.latent_dim * 4) # PositionwiseFeedForward hid dim
        self.dropout = cfg.model.get('dropout', 0.1)

        # Masking parameters
        self.mask_training = cfg.get('mask_training', False)
        self.mask_num = cfg.get('mask_num', 2)  

        # --- 改进的IMU编码器 ---
        # 人体IMU编码器 - 处理6个IMU传感器
        self.human_imu_encoder = IMUEncoder(
            acc_dim=3, 
            ori_dim=6, 
            hidden_dim=64, 
            output_dim=32, 
            dropout=self.dropout
        )
        
        # 物体IMU编码器 - 处理1个IMU传感器
        self.obj_imu_encoder = IMUEncoder(
            acc_dim=3, 
            ori_dim=6, 
            hidden_dim=64, 
            output_dim=32, 
            dropout=self.dropout
        )
        
        # 各IMU传感器之间的注意力机制 (可选，但可以帮助捕捉IMU之间的关系)
        self.imu_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 特征聚合
        num_total_imus_features = 32 * 6 + 32 * 1  # 6个人体IMU + 1个物体IMU
        self.imu_condition_encoder = nn.Sequential(
            nn.Linear(num_total_imus_features, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout)
        )
        
        # Target Data Projection (projects target pose/obj data to latent_dim)
        self.target_feature_dim = 138 # 132 (motion) + 6 (obj_rot)
        self.input_projection = nn.Linear(self.target_feature_dim, self.latent_dim)

        # --- Timestep Embedding ---
        self.embed_timestep = TimestepEmbedder(self.latent_dim)

        # --- Denoiser Network ---
        self.denoiser = OMOMODenoiserWithInput(
            seq_len=input_length,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        # --- Output Projection ---
        # Projects the denoised latent state back to the target data dimension
        self.output_projection = nn.Linear(self.latent_dim, self.target_feature_dim)

    def _encode_imu(self, human_imu, obj_imu):
        """ 
        编码IMU数据并投影到latent_dim。
        
        参数:
            human_imu: 人体IMU数据 [bs, seq, num_imus, 9] (9D = 加速度3D + 6D旋转表示)
            obj_imu: 物体IMU数据 [bs, seq, 1, 9] (9D = 加速度3D + 6D旋转表示)
            
        返回:
            cond_emb: 条件嵌入 [bs, seq, latent_dim]
        """
        bs, seq, num_imus, _ = human_imu.shape
        
        # 编码人体IMU - 先打平处理每个IMU
        human_imu_flat = human_imu.reshape(bs * seq * num_imus, -1)
        human_imu_features = self.human_imu_encoder(human_imu_flat)
        human_imu_features = human_imu_features.reshape(bs, seq, num_imus, -1)
        
        # 应用IMU间注意力机制 (可选)
        if hasattr(self, 'imu_attention'):
            # 对每个时间步应用注意力
            refined_features = []
            for t in range(seq):
                # [bs, num_imus, hidden_dim]
                time_features = human_imu_features[:, t]
                # 自注意力: 每个IMU关注其他IMU
                attn_out, _ = self.imu_attention(time_features, time_features, time_features)
                refined_features.append(attn_out)
            # 重新组装时序数据
            human_imu_features = torch.stack(refined_features, dim=1)
        
        # 打平为每个序列的特征向量
        human_imu_features = human_imu_features.reshape(bs, seq, -1)
        
        # 应用遮蔽（如果训练阶段）
        if self.training and self.mask_training:
            # 重塑为每个IMU分开的特征
            human_imu_features_reshaped = human_imu_features.reshape(bs, seq, num_imus, -1)
            for i in range(bs):
                mask_index = torch.randint(0, num_imus, (self.mask_num,), device=human_imu.device)
                human_imu_features_reshaped[i, :, mask_index] = 0.01  # 遮蔽值
            human_imu_features = human_imu_features_reshaped.reshape(bs, seq, -1)
        
        # 编码物体IMU
        obj_imu_flat = obj_imu.reshape(bs * seq, -1)
        obj_imu_features = self.obj_imu_encoder(obj_imu_flat)
        obj_imu_features = obj_imu_features.reshape(bs, seq, -1)
        
        # 组合并投影
        combined_features = torch.cat([human_imu_features, obj_imu_features], dim=-1)
        cond_emb = self.imu_condition_encoder(combined_features)  # [bs, seq, latent_dim]
        
        return cond_emb

    def diffusion_reverse(self, data=None):
        if data is None: raise ValueError("Input data cannot be empty")
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        device = human_imu.device
        bs, seq = human_imu.shape[:2]

        # 1. Encode Condition
        cond_emb = self._encode_imu(human_imu, obj_imu) # [bs, seq, latent_dim]

        # 2. Initialize Latents (Noise)
        latents = torch.randn((bs, seq, self.latent_dim), device=device).float()
        init_sigma = self.cfg.get('init_noise_sigma', self.scheduler.init_noise_sigma)
        latents = latents * init_sigma

        # 3. Prepare Scheduler and Timesteps
        self.scheduler.set_timesteps(self.cfg.scheduler.num_inference_timesteps)
        timesteps_list = self.scheduler.timesteps.to(device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.scheduler.get("eta", 0.0)

        # 4. Denoising Loop
        for t in timesteps_list:
            timestep_emb = self.embed_timestep(t.expand(bs)) # [bs, latent_dim]
            
            # Call the OMOMO-style denoiser
            # It needs noisy_latents, timestep_emb, cond_emb
            # The OMOMODecoder's forward now primarily processes cond_emb and timestep_emb
            # It doesn't directly take the noisy latents as input in the same way as the previous TransformerDenoiser
            
            # --- Re-alignment needed ---
            # The OMOMO model structure used in `transformer_hand_foot_manip_cond_diffusion_model.py`
            # *predicts* the target `x0` directly (`objective='pred_x0'`).
            # Its denoise_fn takes the *noisy target data* (`x` in `p_losses`) as input, along with time and condition.
            # Let's adjust our `OMOMODenoiser` and the call here to match that.
            
            # `OMOMODecoder` needs to be adjusted to accept `x_t_latent` as input as well.
            # Let's revert `OMOMODecoder` structure slightly to be closer to a denoiser.

            # --- Revised Approach: Denoiser takes x_t, time, cond ---
            # We need a denoiser that takes x_t (noisy latents), time, and condition.
            # Let's rename OMOMODecoder back to TransformerDenoiser but use OMOMO's layers.
            
            # predicted_latent is the output of the denoiser network (e.g., predicted noise or x0 latent)
            model_output = self.denoiser(noisy_latents=latents, timestep_emb=timestep_emb, cond_emb=cond_emb) # Pass arguments clearly
            
            # Process model_output based on scheduler's prediction type
            prediction_type = self.scheduler.config.prediction_type
            if prediction_type == "epsilon":
                latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
            elif prediction_type == "sample": # Assumes model_output is predicted x0 latent
                latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
            elif prediction_type == "v_prediction": # Assumes model_output is predicted v
                latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
            else:
                 raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        # 5. Project Back to Target Dimension
        final_output = self.output_projection(latents) # [bs, seq, 138]
        
        # 6. 分离输出组件
        motion_pred = final_output[:, :, :132]
        obj_rot_pred = final_output[:, :, 132:138]  # 直接取6D旋转表示，不需要reshape

        return {"motion": motion_pred, "obj_rot": obj_rot_pred}

    def forward(self, data=None):
        if data is None: raise ValueError("Input data cannot be empty")
        # Extract GT data
        root_pos = data["root_pos"]
        motion = data["motion"]
        human_imu = data["human_imu"]
        obj_imu = data["obj_imu"]
        obj_trans = data["obj_trans"]
        obj_rot = data["obj_rot"]  # 现在是 [bs, seq, 6] - 6D旋转表示
        device = human_imu.device
        bs, seq = human_imu.shape[:2]

        # 1. Prepare Target Latents
        # obj_rot不再需要reshape，因为已经是扁平的6D表示
        # 目标向量维度现在是 138 (132 + 6)
        target = torch.cat([motion, obj_rot], dim=-1)
        target_latents = self.input_projection(target) # [bs, seq, latent_dim]

        # 2. Encode Condition
        cond_emb = self._encode_imu(human_imu, obj_imu) # [bs, seq, latent_dim]

        # 3. Prepare Noise and Timesteps
        noise = torch.randn_like(target_latents).float()
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=device).long()
        
        # 4. Create Noisy Latents
        noisy_target_latents = self.scheduler.add_noise(target_latents, noise, timesteps) # [bs, seq, latent_dim]

        # 5. Get Timestep Embedding
        timestep_emb = self.embed_timestep(timesteps) # [bs, latent_dim]

        # 6. Denoise using the Transformer-based Denoiser
        # Pass noisy latents, timestep embedding, and condition embedding
        model_output = self.denoiser(noisy_latents=noisy_target_latents, timestep_emb=timestep_emb, cond_emb=cond_emb)

        # 7. Calculate Predicted Noise (consistent with previous logic)
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            predicted_noise = model_output
        elif prediction_type == "sample": 
             alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].to(device).view(-1, 1, 1)
             beta_prod_t = 1 - alpha_prod_t
             predicted_noise = (noisy_target_latents - torch.sqrt(alpha_prod_t) * model_output) / torch.sqrt(beta_prod_t)
        elif prediction_type == "v_prediction":
             alpha_prod_t = self.scheduler.alphas_cumprod[timesteps].to(device).view(-1, 1, 1)
             beta_prod_t = 1 - alpha_prod_t
             predicted_noise = torch.sqrt(alpha_prod_t) * model_output + torch.sqrt(beta_prod_t) * noisy_target_latents
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        return predicted_noise, noise

# Need to redefine the Denoiser using OMOMO's layers but accepting x_t, time, cond
class OMOMODenoiserWithInput(nn.Module):
    """
    Revised Denoiser using OMOMO layers, but accepting noisy input (x_t) 
    alongside time and condition, more like a standard denoiser.
    """
    def __init__(self, seq_len, latent_dim, num_layers, num_heads, d_k, d_v, d_ff, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input projection for x_t might be needed if we combine it differently
        # self.xt_proj = nn.Linear(latent_dim, latent_dim) 
        
        # Timestep embedding layer
        self.embed_timestep = TimestepEmbedder(latent_dim) 
        
        # Condition projection layer
        self.cond_proj = nn.Linear(latent_dim, latent_dim)

        # Positional encoding (fixed, non-trainable)
        # OMOMO Decoder uses table size max_timesteps+1. Adjust if needed.
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(seq_len + 1, latent_dim, padding_idx=0), 
            freeze=True)

        # OMOMO Decoder Layers
        self.layer_stack = nn.ModuleList([
            DecoderLayer(latent_dim, num_heads, d_k, d_v, d_ff, dropout=dropout)
            for _ in range(num_layers)])
        
        self.layer_norm = nn.LayerNorm(latent_dim) # Final LayerNorm

    def forward(self, noisy_latents, timestep_emb, cond_emb):
        # noisy_latents: BS X T X D
        # timestep_emb: BS X D (Needs unsqueezing)
        # cond_emb: BS X T X D
        
        bs, seq_len, d_model = noisy_latents.shape

        # 1. Project condition
        cond_projected = self.cond_proj(cond_emb) # BS x T x D

        # 2. Create sequence positions [0, 1, ..., T-1]
        seq_positions = torch.arange(seq_len, device=noisy_latents.device).unsqueeze(0).expand(bs, -1) # BS x T
        
        # 3. Get positional embeddings
        pos_emb = self.position_enc(seq_positions) # BS x T x D

        # 4. Prepare input: Combine noisy latents, pos embedding, time embedding, and condition
        #    How to combine is key. 
        #    Option A (Standard Transformer): Input = x_t + pos + time. Condition = memory.
        #    Option B (OMOMO-like): Input = cond + pos + time? This seems less like denoising.
        #    Let's try Option A but use OMOMO's DecoderLayer (which lacks cross-attention).
        #    This means the condition must be fused *before* the layers. Add cond to input.
        
        input_sequence = noisy_latents + pos_emb + timestep_emb.unsqueeze(1) + cond_projected
        
        # --- Self-Attention Masking ---
        slf_attn_mask = None # No subsequent mask for standard denoising

        # --- Pass through Decoder Layers ---
        dec_output = input_sequence
        for dec_layer in self.layer_stack:
            # OMOMO DecoderLayer only takes input and self-attn mask
            dec_output, _ = dec_layer(
                dec_output, 
                slf_attn_mask=slf_attn_mask, 
            )

        # Final LayerNorm
        dec_output = self.layer_norm(dec_output)

        return dec_output # Output is predicted noise latent or x0 latent 