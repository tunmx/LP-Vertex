from breezevertex.model import build_model

net = build_model("MobileVertex", width_mult=0.5)

print(net)