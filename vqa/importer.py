def get_model_types():
    model_types = [
        'vqa', 'standard', 'standard_testmask',
        'vlmap_only', 'vlmap_finetune', 'vlmap_answer',
        'vlmap_answer_adapt',
        'vlmap_answer_full', 'vlmap_answer_no_noise'
    ]
    return model_types


def get_model_class(model_type='vqa'):
    if model_type == 'vqa':
        from vqa.model_vqa import Model
    elif model_type == 'standard':
        from vqa.model_standard import Model
    elif model_type == 'standard_testmask':
        from vqa.model_standard_testmask import Model
    elif model_type == 'vlmap_only':
        from vqa.model_vlmap_only import Model
    elif model_type == 'vlmap_finetune':
        from vqa.model_vlmap_finetune import Model
    elif model_type == 'vlmap_answer':
        from vqa.model_vlmap_answer import Model
    elif model_type == 'vlmap_answer_adapt':
        from vqa.model_vlmap_answer_adapt import Model
    elif model_type == 'vlmap_answer_full':
        from vqa.model_vlmap_answer_full import Model
    elif model_type == 'vlmap_answer_no_noise':
        from vqa.model_vlmap_answer_no_noise import Model
    else:
        raise ValueError('Unknown model_type')
    return Model
