def get_model_types():
    model_types = [
        'vqa',
        'standard', 'standard_testmask', 'standard_word2vec',
        'vlmap_only', 'vlmap_finetune', 'vlmap_answer',
        'vlmap_answer_vqa_all',
        'vlmap_answer_vqa_all2',
        'vlmap_answer2',
        'vlmap_answer_noc',
        'vlmap_answer_nocarch',
        'vlmap_answer_adapt', 'vlmap_answer_ent',
        'vlmap_answer_full', 'vlmap_answer_no_noise'
    ]
    return model_types


def get_model_class(model_type='vqa'):
    if model_type == 'vqa':
        from vqa.model_vqa import Model
    elif model_type == 'standard':
        from vqa.model_standard import Model
    elif model_type == 'standard_word2vec':
        from vqa.model_standard_word2vec import Model
    elif model_type == 'standard_testmask':
        from vqa.model_standard_testmask import Model
    elif model_type == 'vlmap_only':
        from vqa.model_vlmap_only import Model
    elif model_type == 'vlmap_finetune':
        from vqa.model_vlmap_finetune import Model
    elif model_type == 'vlmap_answer':
        from vqa.model_vlmap_answer import Model
    elif model_type == 'vlmap_answer_vqa_all':
        from vqa.model_vlmap_answer_vqa_all import Model
    elif model_type == 'vlmap_answer_vqa_all2':
        from vqa.model_vlmap_answer_vqa_all2 import Model
    elif model_type == 'vlmap_answer2':
        from vqa.model_vlmap_answer2 import Model
    elif model_type == 'vlmap_answer_noc':
        from vqa.model_vlmap_answer_noc import Model
    elif model_type == 'vlmap_answer_nocarch':
        from vqa.model_vlmap_answer_nocarch import Model
    elif model_type == 'vlmap_answer_ent':
        from vqa.model_vlmap_answer_ent import Model
    elif model_type == 'vlmap_answer_adapt':
        from vqa.model_vlmap_answer_adapt import Model
    elif model_type == 'vlmap_answer_full':
        from vqa.model_vlmap_answer_full import Model
    elif model_type == 'vlmap_answer_no_noise':
        from vqa.model_vlmap_answer_no_noise import Model
    else:
        raise ValueError('Unknown model_type: {}'.format(model_type))
    return Model
