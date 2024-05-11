import torch
from ultralytics.nn.modules.block import HGBlock, C2f, RepNCSPELAN4, DCNv3C2f
from ultralytics.utils.torch_utils import profile

c2f = C2f(256, 256, 1)
cspdcn = DCNv3C2f(256, 256, 1)
gelan = RepNCSPELAN4(256, 256, 128, 64, 1)
hg = HGBlock(256, 128, 256, 5, 2, True)

# The first two set of data is not included in the calculation due to instability caused by my equipment
results = profile(input=torch.randn(16, 256, 20, 20),
                  ops=[c2f, cspdcn, gelan, hg,
                       c2f, cspdcn, gelan, hg,
                       # train mode for forward and backward latecy
                       c2f, cspdcn, gelan, hg,
                       c2f, cspdcn, gelan, hg,
                       c2f, cspdcn, gelan, hg,
                       c2f, cspdcn, gelan, hg,
                       c2f, cspdcn, gelan, hg,
                       # eval mode for forward latecy
                       # I don't think anyone is interested in backward latecy when eval
                       c2f.eval(), cspdcn.eval(), gelan.eval(), hg.eval(),
                       c2f.eval(), cspdcn.eval(), gelan.eval(), hg.eval(),
                       c2f.eval(), cspdcn.eval(), gelan.eval(), hg.eval(),
                       c2f.eval(), cspdcn.eval(), gelan.eval(), hg.eval(),
                       c2f.eval(), cspdcn.eval(), gelan.eval(), hg.eval()])


