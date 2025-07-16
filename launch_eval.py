from argparse import ArgumentParser
import subprocess
import os, sys
import yaml



def lighteval(config):
    """
        Construct the lighteval command generic to all tasks/custom_tasks
    """
    model = config["model"] 
    TASKS  = config["tasks"]
    CUSTOM_TASKS  = config["custom_tasks"] 
    OUTPUT_DIR  = config["OUTPUT_DIR"]
        
    model_args = f"pretrained={model},dtype={config['precision']},trust_remote_code={config.get('trust_remote_code', False)}"
    accelerate_args = ""
    if  config['DP'] > 1:
        accelerate_args += f"--multi_gpu --num_processes={config['DP']} -m "
    else:
        accelerate_args += f"--num_processes=1 -m "
        
    template = "--use-chat-template " if config['chat_template'] else ""
    model_parallel = ",model_parallel=False " if config['MP'] == 1 else ",model_parallel=True "
    max_samples = f"--max-samples {config.get('max_samples')}" if config.get('max_samples') else ""
    save_details = f"--save-details " if config.get('save_details') else ""
    disable_thinking = f"--disable-thinking " if config.get('disable_thinking') else ""
    
    command = (
        f"accelerate launch {accelerate_args} lighteval accelerate "
        f"{model_args}{model_parallel} "
        f"'{TASKS}' "
        f"--override-batch-size 1 "
        f"--output-dir {OUTPUT_DIR} "
        f"--custom-tasks {CUSTOM_TASKS} "
        f"{template} "
        f"{save_details} "
        f"{max_samples} "
        f"{disable_thinking} "
    )
    
    return command


def evalplus(config):
    """
        Construct the evalplus command 
    """
    pass
    


def eval_local(command):       
    subprocess.run([f"{command}"], shell=True ,check=True)

eval_mapper = {
    "lighteval": lighteval,
    "evalplus": evalplus,
}

def main(args):
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    # eval_func = eval_mapper.get(args.eval)
    eval_func = config["backend"]
    if eval_func:
        command = eval_func(config)
    else:
        print("the passed eval is not defined")
        os.exit(0)
        
    print(command)
    eval_local(command)




if __name__ == '__main__':
    
    parser = ArgumentParser(description="Evaluation models on 3LM benchmarks")
    parser.add_argument('-c', '--config', help='path for yaml eval config')
    
    args = parser.parse_args()
    
    if args.options:
        print("Available Eval functions\n#########################")
        for k, _ in eval_mapper.items():
            print(k)
        os.exit(0)
    
    main(args)