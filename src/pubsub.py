from google.cloud import pubsub_v1

project_id = "centered-sight"

topic_id = "yhs_data"


subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)


def callback(message):
    print(f"Received message: {message.data.decode('utf-8')}")
    print(f"Attributes: {message.attributes}")
    # 메시지 처리가 완료되면 승인(ack)합니다
    message.ack()


streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)