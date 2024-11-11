import core.register as register

__all__ = [
    # register
    'generator_register',
    'discriminator_register',
    'loss_register',
    'optimizer_register',
    'scheduler_register',
    'pipeline_register',
    'layer_register',
    'metric_register',
    'dataset_register'
]

generator_register = register.GeneratorRegister()
discriminator_register = register.DiscriminatorRegister()
loss_register = register.LossRegister()
optimizer_register = register.OptimizerRegister()
scheduler_register = register.SchedulerRegister()
pipeline_register = register.PipelineRegister()
layer_register = register.LayerRegister()
metric_register = register.MetricRegister()
dataset_register = register.DatasetRegister()

del register