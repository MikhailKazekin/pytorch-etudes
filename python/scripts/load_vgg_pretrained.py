import json

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import VGG
from torchvision.models.resnet import resnet50, resnet152
from torchvision.models.vgg import make_layers, cfg, vgg13, vgg16

name_to_scheme = {
    'vgg19': cfg['E'],
    'vgg16': cfg['D']
}


def vgg_from_file(state_dict, name='vgg16', pretrained=False, **kwargs):
    model = VGG(make_layers(name_to_scheme[name]), **kwargs)

    if pretrained:
        model.load_state_dict(state_dict)
    return model


def load_vgg():
    return vgg13(True)


def load_resnet(**kwargs):
    return resnet152(True)


if __name__ == '__main__':

    ft = torch.FloatTensor()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transf = transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    img = Image.open('/home/kazmikh/Projects/mask_rcnn/data/RbZqoQs.jpg')

    tz = transf(img)
    tz = tz.unsqueeze_(0)

    inp = Variable(tz, volatile=True)
    # inp = inp.cuda()

    # model = load_vgg()
    model = load_resnet()
    # model.cuda()

    with open('/home/kazmikh/Projects/mask_rcnn/data/labels.json') as lbl_fp:
        cls_labels = json.load(lbl_fp)

    cls = model.forward(inp)

    cls_np = cls.data.numpy()

    top_idx = cls_np.argsort()

    top_5 = reversed(list(top_idx[0,-5:]))
    for cls_id in top_5:
        print("{0}: {1}".format(cls_labels[str(cls_id)], cls_np[0, cls_id]))

    # top_cls = cls_np.argmax()
    # print("{0}: {1}".format(cls_labels[str(top_cls)], cls_np[0, top_cls]))
    #
    # exit()
