import torch
from copy import copy


rtk_classnames = {
    0: 'background', 1: 'roadAsphalt', 2: 'roadPaved', 3: 'roadUnpaved', 4: 'roadMarking',
    5: 'speedBump', 6: 'catsEye', 7: 'stormDrain', 8: 'manholeCover', 9: 'patch',
    10: 'waterPuddle', 11: 'pothole', 12: 'crack'
}

rtk_sign_labels = torch.Tensor([4, 5, 6])

mocamba_classnames = {
    0: 'Background', 1: 'Animals', 2: 'Asphalt', 3: 'Cat\'s eyes', 4: 'Cracks', 5: 'Dirt road', 6: 'Ego',
    7: 'Hard sand', 8: 'Markings', 9: 'Obstacles', 10: 'People', 11: 'Potholes', 12: 'Retaining walls',
    13: 'Soft sand', 14: 'Speed bump', 15: 'Vehicles', 16: 'Wet sand'
}

mocamba_cats = {
    0: 'Background', 1: 'Things', 2: 'Surfaces', 3: 'Signs', 4: 'Damages', 5: 'Surfaces', 6: 'Things',
    7: 'Surfaces', 8: 'Signs', 9: 'Things', 10: 'Things', 11: 'Damages', 12: 'Signs',
    13: 'Surfaces', 14: 'Signs', 15: 'Things', 16: 'Surfaces'
}

shift = len(mocamba_classnames) - 1
rtk2mocamba = torch.Tensor([
    0, 2, shift+1, 5, 8, 14, 3, shift+2, shift+3, shift+4, shift+5, shift+6, 4]).long()

rtk2mocamba_classnames = {
    shift+1: 'roadPaved', shift+2: 'stormDrain', shift+3: 'manholeCover', shift+4: 'patch',
    shift+5: 'waterPuddle', shift+6: 'pothole'
}

surfaces_ix = [2, 5, 7, 13, 16]
surfaces_names = list(map(mocamba_classnames.get, surfaces_ix))
surfaces_dict = {cl_ix:i for i, cl_ix in enumerate(surfaces_ix)}
surfaces_dict_reverse = {v:k for k, v in surfaces_dict.items()}


class IDs:
    def __init__(self, ds_name='mocamba'):
        self.names_possible = ['mocamba', 'mocamba+rtk']
        assert ds_name in self.names_possible, 'Dataset name not recognized'

        self.id2label = copy(mocamba_classnames)
        if ds_name == 'mocamba+rtk':
            self.id2label.update(rtk2mocamba_classnames)

        self.n_classes = len(self.id2label)
