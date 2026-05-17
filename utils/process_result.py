import json
import os
from pathlib import Path


def get_required_env_vars(required_vars):
    """Load and validate required environment variables."""
    env_values = {}
    missing_env_vars = []

    for var_name in required_vars:
        value = os.environ.get(var_name)
        if value is None:
            missing_env_vars.append(var_name)
        env_values[var_name] = value

    if missing_env_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}")

    return env_values


# Base required env vars
base_env = get_required_env_vars([
    'RUNNER_TYPE', 'FRAMEWORK', 'PRECISION', 'SPEC_DECODING',
    'RESULT_FILENAME', 'ISL', 'OSL', 'DISAGG', 'MODEL_PREFIX', 'IMAGE'
])

hw = base_env['RUNNER_TYPE']
model_prefix = base_env['MODEL_PREFIX']
framework = base_env['FRAMEWORK']
precision = base_env['PRECISION']
spec_decoding = base_env['SPEC_DECODING']
disagg = base_env['DISAGG'].lower() == 'true'
result_filename = base_env['RESULT_FILENAME']
isl = base_env['ISL']
osl = base_env['OSL']
image = base_env['IMAGE']

with open(f'{result_filename}.json') as f:
    bmk_result = json.load(f)


def _build_vllm_omni_data(bmk_result):
    single_node_env = get_required_env_vars(['TP'])
    tp_size = int(single_node_env['TP'])

    images_per_sec = float(bmk_result.get('throughput_qps', 0))
    latency_value = bmk_result.get('latency_per_sample', bmk_result.get('latency_p50'))
    task = bmk_result.get('task')
    modality = 'video' if task in {'t2v', 'i2v', 'ti2v'} else 'image'
    throughput_per_gpu = images_per_sec / tp_size if tp_size > 0 else images_per_sec

    output_shape = {}
    if bmk_result.get('width') is not None:
        output_shape['width'] = bmk_result.get('width')
    if bmk_result.get('height') is not None:
        output_shape['height'] = bmk_result.get('height')

    workload_params = {}
    if bmk_result.get('num_inference_steps') is not None:
        workload_params['num_inference_steps'] = bmk_result.get('num_inference_steps')
    if bmk_result.get('num_frames') is not None:
        workload_params['num_frames'] = bmk_result.get('num_frames')
    if bmk_result.get('fps') is not None:
        workload_params['fps'] = bmk_result.get('fps')
    if bmk_result.get('seed') is not None:
        workload_params['seed'] = bmk_result.get('seed')
    if task is not None:
        workload_params['task'] = task
    if bmk_result.get('dataset') is not None:
        workload_params['dataset'] = bmk_result.get('dataset')

    data = {
        'hw': hw,
        'conc': int(bmk_result.get('max_concurrency', 0)),
        'image': image,
        'model': bmk_result.get('model_id', ''),
        'infmax_model_prefix': model_prefix,
        'framework': framework,
        'precision': precision,
        'spec_decoding': spec_decoding,
        'disagg': disagg,
        # Diffusion/image benchmarks do not have text sequence lengths; keep schema placeholders at 0.
        'isl': 0,
        'osl': 0,
        'is_multinode': False,
        'tp': tp_size,
        'ep': 1,
        'dp_attention': 'false',
        'modality': modality,
        'workload_family': 'diffusion',
        'output_shape': output_shape,
        'workload_params': workload_params,
        'throughput': images_per_sec,
        'throughput_per_gpu': throughput_per_gpu,
        'tput_per_gpu': throughput_per_gpu,
        'output_tput_per_gpu': throughput_per_gpu,
        'input_tput_per_gpu': 0,
        'throughput_unit': 'samples/s',
        'latency_unit': 's/sample',
    }
    if latency_value is not None:
        data['latency'] = latency_value

    # Keep the aggregated diffusion schema compact: width/height live under
    # output_shape, request config lives under workload_params, and framework
    # already captures the backend family.
    excluded_keys = {
        'throughput_qps',
        'max_concurrency',
        'model_id',
        'backend',
        'width',
        'height',
    }
    data.update({k: v for k, v in bmk_result.items() if k not in excluded_keys})
    return data


if framework == 'vllm-omni':
    data = _build_vllm_omni_data(bmk_result)
else:
    data = {
        'hw': hw,
        'conc': int(bmk_result['max_concurrency']),
        'image': image,
        'model': bmk_result['model_id'],
        'infmax_model_prefix': model_prefix,
        'framework': framework,
        'precision': precision,
        'spec_decoding': spec_decoding,
        'disagg': disagg,
        'isl': int(isl),
        'osl': int(osl),
    }

    is_multinode = os.environ.get('IS_MULTINODE', 'false').lower() == 'true'

    if is_multinode:
        # TODO: Eventually will have to have a separate condition in here for multinode disagg and
        # multinode agg. For now, just assume that multinode implies disagg.

        multinode_env = get_required_env_vars(['PREFILL_GPUS', 'DECODE_GPUS', 'PREFILL_NUM_WORKERS', 'PREFILL_TP',
                                              'PREFILL_EP', 'PREFILL_DP_ATTN', 'DECODE_NUM_WORKERS', 'DECODE_TP', 'DECODE_EP', 'DECODE_DP_ATTN'])
        prefill_gpus = int(multinode_env['PREFILL_GPUS'])
        decode_gpus = int(multinode_env['DECODE_GPUS'])
        prefill_num_workers = int(multinode_env['PREFILL_NUM_WORKERS'])
        prefill_tp = int(multinode_env['PREFILL_TP'])
        prefill_ep = int(multinode_env['PREFILL_EP'])
        prefill_dp_attn = multinode_env['PREFILL_DP_ATTN']
        decode_num_workers = int(multinode_env['DECODE_NUM_WORKERS'])
        decode_tp = int(multinode_env['DECODE_TP'])
        decode_ep = int(multinode_env['DECODE_EP'])
        decode_dp_attn = multinode_env['DECODE_DP_ATTN']

        total_gpus = prefill_gpus + decode_gpus
        if total_gpus <= 0:
            raise ValueError("Multinode results require at least one GPU.")
        if prefill_gpus <= 0:
            raise ValueError("Multinode results require at least one prefill GPU.")

        output_tput_denominator = decode_gpus if decode_gpus > 0 else total_gpus
        output_decode_tp = decode_tp if decode_gpus > 0 else 0
        output_decode_ep = decode_ep if decode_gpus > 0 else 0

        multi_node_data = {
            'is_multinode': True,
            'prefill_tp': prefill_tp,
            'prefill_ep': prefill_ep,
            'prefill_dp_attention': prefill_dp_attn,
            'prefill_num_workers': prefill_num_workers,
            'decode_tp': output_decode_tp,
            'decode_ep': output_decode_ep,
            'decode_dp_attention': decode_dp_attn,
            'decode_num_workers': decode_num_workers,
            'num_prefill_gpu': prefill_gpus,
            'num_decode_gpu': decode_gpus,
            'tput_per_gpu': float(bmk_result['total_token_throughput']) / total_gpus,
            'output_tput_per_gpu': float(bmk_result['output_throughput']) / output_tput_denominator,
            'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / prefill_gpus,
        }

        data = data | multi_node_data
    else:
        if disagg:
            raise ValueError("Disaggregated mode requires multinode setup.")

        single_node_env = get_required_env_vars(['TP', 'EP_SIZE', 'DP_ATTENTION'])
        tp_size = int(single_node_env['TP'])
        ep_size = int(single_node_env['EP_SIZE'])
        dp_attention = single_node_env['DP_ATTENTION']

        single_node_data = {
            'is_multinode': False,
            'tp': tp_size,
            'ep': ep_size,
            'dp_attention': dp_attention,
            'tput_per_gpu': float(bmk_result['total_token_throughput']) / tp_size,
            'output_tput_per_gpu': float(bmk_result['output_throughput']) / tp_size,
            'input_tput_per_gpu': (float(bmk_result['total_token_throughput']) - float(bmk_result['output_throughput'])) / tp_size,
        }

        data = data | single_node_data

for key, value in bmk_result.items():
    if key.endswith('ms'):
        data[key.replace('_ms', '')] = float(value) / 1000.0
    if 'tpot' in key:
        data[key.replace('_ms', '').replace(
            'tpot', 'intvty')] = 1000.0 / float(value)

print(json.dumps(data, indent=2))

with open(f'agg_{result_filename}.json', 'w') as f:
    json.dump(data, f, indent=2)
