import ml_collections

model = 'RGBD_seg'

if model == 'RGBD_seg':
    base_blocks = [3, 4, 18, 3]
    fe_blocks = [3, 4, 12, 3]
    decoder_channel = 512
    pretrain_weight_dir = r"./pretrain_weight/pvt_v2_b3.pth"

def get_config():
    config = ml_collections.ConfigDict()
    config.image_h = 480
    config.image_w = 640
    config.classes = 40
    config.num_stages = 4
    config.train_file = r"../dataset/NYUv2/train.txt"
    config.val_file = r"../dataset/NYUv2/test.txt"
    config.pretrain_weight_dir = pretrain_weight_dir
    config.train_batch_size = 4
    config.val_batch_size = 1
    config.num_workers = 4
    config.begin_epoch = 0
    config.stop_epoch = 300
    config.save_freq = 10

    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.lr = 0.00006
    config.optimizer.wd = 0.01
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.power = 0.9
    config.lr_scheduler.warm_up_epoch = 5

    # TokenEmbed config
    config.embed = ml_collections.ConfigDict()
    config.embed.channels = [3, 64, 128, 320, 512]
    config.embed.kernel_size = [7, 3, 3, 3]
    config.embed.stride = [4, 2, 2, 2]
    config.embed.padding = [3, 1, 1, 1]

    # Transformer Block config
    config.trans = ml_collections.ConfigDict()
    config.trans.dims = [64, 128, 320, 512]
    config.trans.blocks = base_blocks
    config.trans.drop_path = 0.1

    # attn
    config.trans.attn = ml_collections.ConfigDict()
    config.trans.attn.dims = [64, 128, 320, 512]
    config.trans.attn.sr_ratios = [8, 4, 2, 1]
    config.trans.attn.num_heads = [1, 2, 5, 8]
    config.trans.attn.qkv_bias = True
    config.trans.attn.attn_drop = 0.0
    config.trans.attn.proj_drop = 0.0
    # mlp
    config.trans.mlp = ml_collections.ConfigDict()
    config.trans.mlp.in_features = [64, 128, 320, 512]
    config.trans.mlp.mlp_ratios = [8, 8, 4, 4]
    config.trans.mlp.hidden_features = [a * b for a, b in zip(config.trans.attn.dims, config.trans.mlp.mlp_ratios)]
    config.trans.mlp.drop = 0.0

    # FE
    config.FE = ml_collections.ConfigDict()
    config.FE.I2V = ml_collections.ConfigDict()
    config.FE.I2V.dims = [64, 128, 320, 512]
    config.FE.I2V.target_size = [(1, 1), (3, 4), (6, 8), (12, 16)]
    config.FE.I2V.spp_dims = sum([x * y for x, y in config.FE.I2V.target_size])
    config.FE.I2V.out_dims = [256, 160, 64, 32]

    config.FE.trans = ml_collections.ConfigDict()
    config.FE.trans.dims = [256, 160, 64, 32]
    config.FE.trans.blocks = fe_blocks
    config.FE.trans.drop_path = 0.1

    # attn
    config.FE.trans.attn = ml_collections.ConfigDict()
    config.FE.trans.attn.dims = [256, 160, 64, 32]
    config.FE.trans.attn.num_heads = [8, 5, 2, 1]
    config.FE.trans.attn.qkv_bias = True
    config.FE.trans.attn.attn_drop = 0.0
    config.FE.trans.attn.proj_drop = 0.0
    # mlp
    config.FE.trans.mlp = ml_collections.ConfigDict()
    config.FE.trans.mlp.in_features = [256, 160, 64, 32]
    config.FE.trans.mlp.mlp_ratios = [4, 4, 8, 8]
    config.FE.trans.mlp.hidden_features = [a * b for a, b in zip(config.FE.trans.attn.dims,
                                                                     config.FE.trans.mlp.mlp_ratios)]
    config.FE.trans.mlp.drop = 0.0

    # GFA
    config.FF = ml_collections.ConfigDict()
    config.FF.trans = ml_collections.ConfigDict()
    config.FF.trans.dims = [64, 128, 320, 512]
    config.FF.trans.drop_path = 0.1

    # attn
    config.FF.trans.attn = ml_collections.ConfigDict()
    config.FF.trans.attn.dims = [64, 128, 320, 512]
    config.FF.trans.attn.sr_ratios = [8, 4, 2, 1]
    config.FF.trans.attn.num_heads = [1, 2, 5, 8]
    config.FF.trans.attn.qkv_bias = True
    config.FF.trans.attn.attn_drop = 0.0
    config.FF.trans.attn.proj_drop = 0.0
    # mlp
    config.FF.trans.mlp = ml_collections.ConfigDict()
    config.FF.trans.mlp.in_features = [64, 128, 320, 512]
    config.FF.trans.mlp.mlp_ratios = [4, 4, 2, 2]
    config.FF.trans.mlp.hidden_features = [a * b for a, b in zip(config.FF.trans.attn.dims,
                                                                  config.FF.trans.mlp.mlp_ratios)]
    config.FF.trans.mlp.drop = 0.0

    # head
    config.SegFormer_head = ml_collections.ConfigDict()
    config.SegFormer_head.embedding_dim = decoder_channel
    config.SegFormer_head.drop_out = 0.0

    config.fpn_head = ml_collections.ConfigDict()
    config.fpn_head.dims = config.trans.dims
    config.fpn_head.channels = decoder_channel
    config.fpn_head.dropout_ratio = 0.1
    config.fpn_head.in_index = [0, 1, 2, 3]
    config.fpn_head.feature_strides = [4, 8, 16, 32]

    return config
