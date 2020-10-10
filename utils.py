def parse_data_config(path):
    options = dict()

    options['gpus'] = '0, 1, 2, 3'
    options['num_workers'] = '0'

    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options