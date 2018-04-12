#####################################################
#
# Run the RBM as a job on Paperspace. Assume that you have the paperspace 
# CLI and have done a paperspace login before to store your credentials
#
# Copyright (c) 2018 christianb93
# Permission is hereby granted, free of charge, to 
# any person obtaining a copy of this software and 
# associated documentation files (the "Software"), 
# to deal in the Software without restriction, 
# including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice 
# shall be included in all copies or substantial 
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY 
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS 
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#####################################################


import paperspace.jobs
from paperspace.login import apikey
import paperspace.config

import requests


#
# Define parameters
#
params = {}
# 
# We want to use GIT, so we use the parameter workspaceFileName
# instead of workspace
#
params['workspaceFileName'] = "git+https://github.com/christianb93/MachineLearning"
params['machineType'] = "K80"
params['command'] = "export MPLBACKEND=AGG ; python3 RBM.py \
                --N=28 --data=MNIST \
                --save=1 \
                --tmpdir=/artifacts \
                --hidden=128 \
                --pattern=256 --batch_size=128 \
                --epochs=40000 \
                --run_samples=1 \
                --sample_size=6,6 \
                --beta=1.0 --sample=200000 \
                --algorithm=PCDTF --precision=32"
params['container'] = 'paperspace/tensorflow-python'
params['project'] = "MachineLearning"
params['dest'] = "/tmp"

#
# Get API key
#
apiKey = apikey()
print("Using API key ", apiKey)
#
# Create the job. We do NOT use the create method as it cannot
# handle the GIT feature, but assemble the request ourselves
#
http_method = 'POST'
path = '/' + 'jobs' + '/' + 'createJob'


r = requests.request(http_method, paperspace.config.CONFIG_HOST + path,
                             headers={'x-api-key': apiKey},
                             params=params, files={})
job = r.json()
    
if 'id' not in job:
    print("Error, could not get jobId")
    print(job)
    exit(1)

jobId = job['id']
print("Started job with jobId ", jobId)
params['jobId']  = jobId

#
# Now poll until the job is complete

if job['state'] == 'Pending':
    print('Waiting for job to run...')
    job = paperspace.jobs.waitfor({'jobId': jobId, 'state': 'Running'})

print("Job is now running")
print("Use the following command to observe its logs: ~/node_modules/.bin/paperspace jobs logs --jobId ", jobId, "--tail")

job = paperspace.jobs.waitfor({'jobId': jobId, 'state': 'Stopped'})
print("Job is complete: ", job)

#
# Finally get artifacts
#
print("Downloading artifacts to directory ", params['dest'])
paperspace.jobs.artifactsGet(params)
