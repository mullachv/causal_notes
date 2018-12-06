import tensorflow as tf
import regularizers as regularizers

def logisticRegression(features, labels, mode, params):
    if params['regularizer'] == 'l1':
        kernel_regularizer = regularizers.l1reg(scale=params['lmbda'])
    elif params['regularizer'] == 'l2':
        kernel_regularizer = regularizers.l2reg(scale=params['lmbda'])
    else:
        kernel_regularizer = None
        
    # define model here
    logits = tf.layers.dense(features[params['key']],
                             params['n_classes'],
                             activation=None,
                             name='weights',
                             use_bias=True,
                             kernel_regularizer=kernel_regularizer)
    probabilities = tf.nn.softmax(logits, axis=-1)
    predicted_classes = tf.one_hot(tf.argmax(probabilities, -1), depth=probabilities.shape[1])
    
    # shortcut out if it's just a prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes,
            'probabilities': probabilities,
            'logits': logits,
            'y_pred': predicted_classes
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # loss + regularization
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    if kernel_regularizer is not None:
        loss += tf.reduce_sum(tf.losses.get_regularization_losses())
    
    # evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    
    # store metrics
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    
    # different modes
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
def linearRegression(features, labels, mode, params):
    if params['regularizer'] == 'l1':
        kernel_regularizer = regularizers.l1reg(scale=params['lmbda'])
    elif params['regularizer'] == 'l2':
        kernel_regularizer = regularizers.l2reg(scale=params['lmbda'])
    elif params['regularizer'] == 'offCenteredL1':
        kernel_regularizer = regularizers.offCenteredL1(scale=params['lmbda'], mu=params['weight_priors'])
    else:
        kernel_regularizer = None
        
    # define model here
    output = tf.layers.dense(features[params['key']],
                             params['n_dims'],
                             activation=None,
                             name='weights',
                             use_bias=True,
                             kernel_regularizer=kernel_regularizer)
    
    # shortcut out if it's just a prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'y_pred': output
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # loss + regularization
    loss = tf.reduce_mean(tf.reduce_sum((labels - output) ** 2, axis=-1))
    
    if kernel_regularizer is not None:
        loss += tf.reduce_sum(tf.losses.get_regularization_losses())
    
    # evaluation metrics
    accuracy = tf.metrics.mean_squared_error(labels=labels, predictions=output, name='mse_op')
    
    # store metrics
    rSquared = 1 - tf.reduce_sum((output - labels) ** 2) / tf.reduce_sum((labels - tf.reduce_mean(labels)) ** 2)
    mRSquared, updateRSquared = tf.metrics.mean(rSquared)
    metrics = {'accuracy': accuracy, 'r_squared': (mRSquared, updateRSquared)}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('r_squared', rSquared)
    
    # different modes
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

