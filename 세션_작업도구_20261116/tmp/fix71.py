src = r'review/plan_v70.md'; dst = r'review/plan_v71.md'
t = open(src, 'rb').read().decode('utf-8')
old = r"딥러닝 GPU 메모리 오류는 같은 배치를 나눠 계산하는 gradient accumulation)"
new = r"딥러닝 GPU 메모리 오류는 같은 배치를 나눠 계산하는 gradient accumulation — BatchNormalization 이 있는 구조(AutoKeras 등)는 나눠 계산하지 않고 GPU 메모리를 확보한 뒤 다시)"
assert t.count(old) == 1
t = t.replace(old, new)
open(dst, 'wb').write(t.encode('utf-8'))
print('ok')
