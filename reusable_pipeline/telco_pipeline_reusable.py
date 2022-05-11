import kfp
from kfp import dsl

@dsl.pipeline(
    name='telco_pipeline_reusable',
    description='Telco Pipeline Reusable Components')
def telco_reusable_pipeline():
    feature_prep_op = kfp.components.load_component_from_url(
        "https://raw.githubusercontent.com/Jabor047/Kubeflow-Pipelines/main/reusable_pipeline/feature_prep.yaml"
    )
    model_training_comp = kfp.components.load_component_from_url(
        "https://raw.githubusercontent.com/Jabor047/Kubeflow-Pipelines/main/reusable_pipeline/model_training.yaml"
    )
    model_training_op = model_training_comp(feature_prep_op().outputs["engagement_data"],
                                            feature_prep_op().outputs["experience_data"]).after(feature_prep_op())

kfp.compiler.Compiler().compile(telco_reusable_pipeline, 'telco_pipeline_reusable.zip')

# client = kfp.Client()
# experiment = client.create_experiment(name='telco_pipeline_reusable')
# run = client.run_pipeline(experiment_id=experiment.id, job_name='telco_pipeline_reusable',
#                           pipeline_package_path='telco_pipeline_reusable.zip')
