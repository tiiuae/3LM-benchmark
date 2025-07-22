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
        
    model_args = f"model_name={model},dtype={config['precision']},trust_remote_code={config.get('trust_remote_code', False)},batch_size={config.get('batch_size', 1)}"
    accelerate_args = ""
    if  config['DP'] > 1:
        accelerate_args += f"--multi_gpu --num_processes={config['DP']} -m "
    else:
        accelerate_args += f"--num_processes=1 -m "
        
    model_parallel = ",model_parallel=False " if config['MP'] == 1 else ",model_parallel=True "
    max_samples = f"--max-samples {config.get('max_samples')}" if config.get('max_samples') else ""
    save_details = f"--save-details " if config.get('save_details') else ""
    template = f"--use-chat-template " if config['chat_template'] else ""
    disable_thinking = f"--disable-thinking " if config['disable_thinking'] else ""
    
    
    command = (
        f"accelerate launch {accelerate_args} lighteval accelerate "
        f"{model_args}{model_parallel} "
        f"'{TASKS}' "
        f"--output-dir {OUTPUT_DIR} "
        f"--custom-tasks {CUSTOM_TASKS} "
        f"{template} "
        f"{disable_thinking} "
        f"{save_details} "
        f"{max_samples} "
    )
    
    return command


def evalplus_arabic(config):

    model = config["model"]
    dataset = config["dataset"]
    backend = config.get("engine")

    tp = f"--tp {config.get('tp')} " if (config.get('tp') and backend == 'vllm') else ""
    dtype = f"--dtype {config.get('dtype')}" if config.get('dtype') else ""
    
    
    greedy = "--greedy" if config.get("greedy") else ""
    force_base_prompt =  f"--force_base_prompt" if not config.get("chat_template") else ""
    enable_thinking = f"--enable_thinking {config.get('thinking_mode')}"  if config.get('thinking_mode') else ""
    trailing_newline = f"--trailing_newline {config.get('trailing_newline')}"  if config.get('trailing_newline') else ""

    command = (
        "evalplus.evaluate "
        f"--model {model} "
        f"--dataset {dataset} "         
        f"--backend {backend} "   
        f"{tp} {dtype} "
        f"{greedy} "
        f"{force_base_prompt} {enable_thinking} {trailing_newline}"

    )


    return command



def eval_local(command):       
    subprocess.run([f"{command}"], shell=True ,check=True)

eval_mapper = {
    "lighteval": lighteval,
    "evalplus_arabic": evalplus_arabic,
}

def main(args):
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    # eval_func = eval_mapper.get(args.eval)
    backend = config["backend"]
    eval_func = eval_mapper.get(backend)
    
    
    if eval_func is None :
        command = eval_mapper(config)
        print("the passed eval is not defined")


    command = eval_func(config)    
    
    print(command)
    eval_local(command)




if __name__ == '__main__':
    
    parser = ArgumentParser(description="Evaluation models on 3LM benchmarks")
    parser.add_argument('-c', '--config', help='path for yaml eval config')
    
    args = parser.parse_args()
    
    # if args.options:
    #     print("Available Eval functions\n#########################")
    #     for k, _ in eval_mapper.items():
    #         print(k)
    #     os.exit(0)
    
    main(args)
