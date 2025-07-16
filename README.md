# 3LM Repo
This repo will be used to store any code related to Falcon Arabic project

## setup
1. create conda env
2. install requirements
3. install both benchmarks (evalplus and lighteval)
4. if Model is hosted in HF, ensure that you are logged in

## Run
** native + synthetic ** 
`python launch_eval.py -c examples/lighteval_3lm.yaml`

** native** 
`python launch_eval.py -c examples/lighteval_native.yaml`

** synthetic ** 
`python launch_eval.py -c examples/lighteval_synthetic.yaml`

** arabic code ** 
`python launch_eval.py -c examples/evalplus_arabic_code.yaml`
