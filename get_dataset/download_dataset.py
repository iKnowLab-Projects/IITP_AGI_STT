import tensorflow_datasets as tfds

# 'bridge' 또는 'bridge_data_msr' 둘 중 하나를 선택하세요
# dataset, info = tfds.load('bridge', split='train', with_info=True)
dataset, info = tfds.load('bridge_data_msr', split='train', with_info=True)

for episode in dataset:
    steps = episode['steps']
    for step in steps:
        instr = step['language_instruction'].numpy().decode('utf-8')
        print(instr)
    # 예시로 첫번째 episode만 처리 후 종료
    break
