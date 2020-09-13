import six
import keras

def get_function_name(o):
    """Utility function to return the model's name
    """
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__

def to_dict_w_opt(model):
    """Serialize a model and add the config of the optimizer and the loss.
    """
    config = dict()
    config_m = model.get_config()
    config['config'] = {
        'class_name': model.__class__.__name__,
        'config': config_m,
    }
    if hasattr(model, 'optimizer'):
        config['optimizer'] = model.optimizer.get_config()
        config['optimizer']['name']=model.optimizer.__class__.__name__
    if hasattr(model, 'loss'):
        if isinstance(model.loss, dict):
            config['loss'] = dict([(k, get_function_name(v))
                                   for k, v in model.loss.items()])
        else:
            config['loss'] = get_function_name(model.loss)

    return config




def model_from_dict_w_opt(model_dict, custom_objects=None):
    """Builds a model from a serialized model using `to_dict_w_opt`
    """
    if custom_objects is None:
        custom_objects = {}
    def convert_custom_objects(obj):
        """Handles custom object lookup.
        # Arguments
            obj: object, dict, or list.
        # Returns
            The same structure, where occurences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                if value in custom_objects:
                    deserialized.append(custom_objects[value])
                else:
                    deserialized.append(value)
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = []
                if isinstance(value, list):
                    for element in value:
                        if element in custom_objects:
                            deserialized[key].append(custom_objects[element])
                        else:
                            deserialized[key].append(element)
                elif value in custom_objects:
                    deserialized[key] = custom_objects[value]
                else:
                    deserialized[key] = value
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    model = keras.models.model_from_config(model_dict['config'],
                              custom_objects=custom_objects)

    if 'optimizer' in model_dict:
        model_name = model_dict['config'].get('class_name')
        # if it has an optimizer, the model is assumed to be compiled
        loss = model_dict.get('loss')
        metrics = model_dict.get('metrics')
        # if a custom loss function is passed replace it in loss
        if loss in custom_objects:
            loss = convert_custom_objects(loss)

        if metrics in custom_objects:
            metrics = convert_custom_objects(metrics)

        optimizer_params = dict([(
            k, v) for k, v in model_dict.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = keras.optimizers.get(optimizer_name).from_config(optimizer_params)


        sample_weight_mode = model_dict.get('sample_weight_mode', None)
        loss_weights = model_dict.get('loss_weights', None)
        model.compile(loss=loss,
                      metrics = metrics,
                      optimizer=optimizer,
                      sample_weight_mode=sample_weight_mode,
                      loss_weights=loss_weights)
    return model