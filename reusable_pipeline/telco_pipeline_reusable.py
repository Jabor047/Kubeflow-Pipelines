import kfp
from kfp import dsl

def feature_prep():
    feature_prep_op = kfp.components.load_component_from_url()
    return feature_prep_op

def model_training(experience_data, engagement_data):
    return dsl.ContainerOp(
        name='model_training',
        image='jabor047/kubeflow-pipeline-reusable-components-model-training:v.0.1',
        arguments=[
            '--experience_data', experience_data,
            '--engagement_data', engagement_data
        ],
        file_outputs={
            'model': '/model_training/satisfaction_kmeans.pkl'
        }
    )

@dsl.pipeline(
    name='telco_pipeline_reusable',
    description='Telco Pipeline Reusable Components')
def telco_reusable_pipeline():
    feature_prep_op = feature_prep()
    model_training_op = model_training(
        experience_data=dsl.InputArgumentPath(feature_prep_op.outputs['experience_data']),
        engagement_data=dsl.InputArgumentPath(feature_prep_op.outputs['engagement_data'])
    ).after(feature_prep_op)

kfp.compiler.Compiler().compile(telco_reusable_pipeline, 'telco_pipeline_reusable.zip')

# client = kfp.Client()
# experiment = client.create_experiment(name='telco_pipeline_reusable')
# run = client.run_pipeline(experiment_id=experiment.id, job_name='telco_pipeline_reusable',
#                           pipeline_package_path='telco_pipeline_reusable.zip')
