import tensorflow as tf

class Distiller(tf.keras.Model):
    """蒸馏器"""

    def __init__(self, teacher, student, **kwargs):
        super(Distiller, self).__init__(**kwargs)
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss,
        distillation_loss,
        alpha=0.1,
        temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss = student_loss
        self.distillation_loss = distillation_loss
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, inputs):
        X, y = inputs
        teacher_pred = self.teacher(X, training=False)
        with tf.GradientTape() as tape:
            student_pred = self.student(X, training=True)
            sloss = self.student_loss(y, student_pred)
            dloss = self.distillation_loss(
                tf.math.softmax(teacher_pred / self.temperature, axis=1),
                tf.math.softmax(student_pred / self.temperature, axis=1)
            )
            loss = self.alpha * sloss + (1 - self.alpha) * dloss

        student_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, student_vars)
        self.optimizer.apply_gradients(zip(gradients, student_vars))
        self.compiled_metrics.update_state(y, student_pred)
        results = {metric.name: metric.result() for metric in self.metrics}
        results.update({"student_loss": sloss, "distillation_loss": dloss})
        return results

    def test_step(self, inputs):
        X, y = inputs
        y_pred = self.student(X, training=False)
        sloss = self.student_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        results = {metric.name: metric.result() for metric in self.metrics}
        results.update({"student_loss": sloss})
        return results
