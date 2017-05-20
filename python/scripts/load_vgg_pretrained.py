import torch
from torch.autograd import Variable
from torchvision.models import VGG
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import make_layers, cfg

name_to_scheme = {
    'vgg19': cfg['E'],
    'vgg16': cfg['D']
}


def vgg_from_file(state_dict, name='vgg16', pretrained=False, **kwargs):
    model = VGG(make_layers(name_to_scheme[name]), **kwargs)

    if pretrained:
        model.load_state_dict(state_dict)
    return model

def load_resnet(**kwargs):
    return resnet18(pretrained=True, **kwargs)

if __name__ == '__main__':

    import numpy as np

    ft = torch.FloatTensor()

    pic_ar = np.load('/home/kazmikh/Projects/mask_rcnn/pic.npy')

    img = torch.from_numpy(pic_ar.transpose((2, 0, 1)))
    tz = img.float().div(255)

    tz = tz.expand([1] + list(tz.size()))
    inp = Variable(tz).cuda()

    print(inp.size())

    #net_name = 'vgg16'
    #cached_file = '/home/kazmikh/Projects/mask_rcnn/{0}.pth'.format(net_name)

    #st_dict = torch.load(cached_file, map_location={'cpu':'cuda:0'})

    # st_dict['classifier.0.weight'] = st_dict['classifier.1.weight'].repeat(1,1)
    # del st_dict['classifier.1.weight']
    # st_dict['classifier.0.bias'] =   st_dict['classifier.1.bias'].repeat(1,1)
    # del st_dict['classifier.1.bias']
    #
    # st_dict['classifier.3.weight'] = st_dict['classifier.4.weight'].repeat(1,1)
    # del st_dict['classifier.4.weight']
    # st_dict['classifier.3.bias'] =   st_dict['classifier.4.bias'].repeat(1,1)
    # del st_dict['classifier.4.bias']

    #for k, v in st_dict.items():
    #    print('{0}\t{1}'.format(k, v.size()))
    #print('\n\n')

    # model = vgg_from_file(name=net_name, state_dict=st_dict, pretrained=True)

    model = load_resnet()
    model.forward(inp)
    print(model.forward(inp))
