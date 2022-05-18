
# Experience and Satisfcation Score Telco Company

This repository contains code that gets experience and satisfaction scores for users using a telecommunication compnay's service. Since the dataset is not labeled I had to employ unsurpervised learning to cluster the users and calculate either scores from the clusters created.

I then set up the project as an end to end pipeline on Kubeflow pipeline.

## folder structure
* lightweight_pipeline - this contains all the code for the project.
    - auth_cookie.md - Contains instruction on how to get the authservice_session_cookie.
    - telco_pipeline.py - contains the code necessary to create the pipeline.
    - telco_pipeline.zip - This is the pipeline packaged as a zip


