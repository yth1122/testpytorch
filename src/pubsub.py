from google.cloud import pubsub_v1
import os
project_id = "centered-sight-237801"

subscription_id = "yhs_python_test"


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/app/key/dev.json'

obj = {'startX':0 ,'endX': 0,'startZ':0,'endZ':0}


def callback(message):
    decode_text = message.data.decode('utf-8')
    select_mkey = '7'
    # ai에 사용될 데이터 저장
    
    if len(decode_text) > 1:
      data = decode_text.split('|')
      if data[0] != 'EXTRA':
        mkey = data[2]
        if mkey == '7':
          for i in range(4,len(data),2):
            if data[i] == 'path_position':
              [x,y,z] = data[i+1].split(' ')
              x, z = float(x), float(z)
              print(f"{x}{y}{z}")
              if obj['startX'] == 0 and obj['startZ'] == 0 :
                print('start')
                obj['startX'] = x
                obj['startZ'] = z
              else:
                obj['endX'] = x
                obj['endZ'] = z
            elif data[i] == 'block':
              print('block',data[i+1],obj)
              obj['startX'] = obj['endX']
              obj['endX'] = 0
              obj['startZ'] = obj['endZ']
              obj['endZ'] = 0
              print(obj)
      elif data[1] == 'PART_COUNT' and data[13] == select_mkey :
        print(f"part_count {data}")   
    # 메시지 처리가 완료되면 승인(ack)합니다
    message.ack()

def subscribe():
    print(os.environ)
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        # 결과를 기다림 (이 경우 무기한 대기)
        streaming_pull_future.result()
    except TimeoutError:
        streaming_pull_future.cancel()  # 구독 취소
        print("Streaming pull future cancelled.")