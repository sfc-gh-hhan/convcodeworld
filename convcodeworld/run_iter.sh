#!/bin/bash
eval "$(conda shell.bash hook)"

CCB_VERSION=v0.3.6
MODEL_NAME=$1
EF=$2
UNIT_TEST=$3
SUF=$4
USER_EXPERTISE=$5

#!/bin/bash
eval "$(conda shell.bash hook)"

model_name=${MODEL_NAME}
version=${CCB_VERSION}
simulator_name=gpt-4o
#save_dir=results
save_dir=/vault/secrets

conda activate bigcodebench
cd bigcodebench
gen_path=sanitized_calibrated_samples/instruct/${model_name//\//--}--bigcodebench-instruct--vllm-0-1-sanitized-calibrated.jsonl
./eval_single_dspy_result.sh $gen_path
cd ..

option="CF"
if [ $EF = 'true' ]; then
  option=${option}"_EF"
  if [ $UNIT_TEST = 'true' ]; then
    option=${option}"_UNIT"
  else
    option=${option}"_FULL"
  fi
fi
if [ $SUF = 'true' ]; then
  if [ $USER_EXPERTISE = "novice" ]; then
    option=${option}"_SNF"
  elif [ $USER_EXPERTISE = "expert" ]; then
    option=${option}"_SEF"
  fi
  if [ $model_name != $simulator_name ]; then
    option=${option}"_by_${simulator_name}"
  fi
fi

for i in {1..10..1}
do
  conda activate ConvCodeBench
  python run.py --model_name ${model_name} --use_generated_code true --generated_code_path bigcodebench/${gen_path} --compilation_feedback true --execution_feedback $EF --unit_test ${UNIT_TEST} --simulated_user_feedback ${SUF} --user_expertise ${USER_EXPERTISE}  --iteration $i --version $version  --save_dir $save_dir

  conda activate bigcodebench
  cd bigcodebench

  gen_path=../results/${version}/bigcodebench_${model_name//\//_}_${option}_ITER=${i}.jsonl
  if [ $i -eq 1 ]; then
    gen_path=../results/${version}/bigcodebench_${model_name//\//_}_${option}.jsonl
  fi
  ./eval_single_dspy_result.sh ${gen_path}
  cd ..
  conda deactivate
  conda deactivate
done