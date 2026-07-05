"""VQ2 state estimation: recorded-corpus replay harness and estimators.

VQ2 removed ODOMETRY/ATTITUDE/LOCAL_POSITION_NED; the estimator is built and
validated OFFLINE against recorded raw-sensor corpora before any live flight
(playbook rule: live flights are for confirmation, never discovery).
"""
