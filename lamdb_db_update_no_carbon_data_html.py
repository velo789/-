import boto3
import csv
import os
import logging
from boto3.dynamodb.conditions import Attr
from decimal import Decimal

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # DynamoDB 및 S3 클라이언트 초기화
    dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')  # 서울 리전
    s3 = boto3.client('s3')
    table_name = 'Policies'
    table = dynamodb.Table(table_name)
    
    # S3 이벤트에서 버킷 이름과 파일 키 가져오기
    for record in event['Records']:

        if record['eventName'].startswith('ObjectCreated:'):  # ObjectCreated 이벤트만 처리
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            
            # 특정 파일 이름 검사
            if 'updated_policies_with_years_and_country_id.csv' in object_key:
                logger.info(f"Processing file: {object_key} from bucket: {bucket_name}")
                
                # S3에서 CSV 파일 다운로드 경로 설정
                download_path = '/tmp/Policies1.csv'
                try:
                    s3.download_file(bucket_name, object_key, download_path)
                    logger.info(f"Downloaded {object_key} to {download_path}")
                except Exception as e:
                    logger.error(f"Error downloading file {object_key} from bucket {bucket_name}: {e}")
                    continue
                
                # CSV 파일에서 데이터를 읽어 DynamoDB에 삽입
                try:
                    with open(download_path, newline='', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile, quotechar='"', quoting=csv.QUOTE_ALL)
                        for row in reader:
                            try:
                                # Year 필드를 정수형으로 변환
                                year = int(row['Year']) if row['Year'] else None
                                
                                # Implementation Cost를 정수형으로 변환
                                implementation_cost = int(Decimal(row['Implementation Cost'])) if row['Implementation Cost'] else None
                                
                                # Estimated Carbon Reduction (%)와 (Million Tons)를 Decimal로 변환
                                estimated_carbon_reduction = Decimal(row['Estimated Carbon Reduction (%)']) if row['Estimated Carbon Reduction (%)'] else None
                                estimated_carbon_reduction_mt = Decimal(row['Estimated Carbon Reduction (Million Tons)']) if row['Estimated Carbon Reduction (Million Tons)'] else None
                                
                                table.put_item(
                                    Item={
                                        'Policy_id': row['Policy_id'],  # CSV에서 직접 Policy_id 사용
                                        'Policy_name': row['Policy'],  # 필드명 수정
                                        'Policy_URL': row['Policy URL'],  # 필드명 수정
                                        'Year': year,  # 정수형으로 변환된 Year 필드
                                        'Jurisdiction': row['Jurisdiction'],
                                        'Content': row['Content'],
                                        'Topics': row['Topics'],
                                        'Policy_Types': row['Policy Types'],  # 필드명 수정
                                        'Sectors': row['Sectors'],
                                        'Implementation_cost': implementation_cost,  # 정수형으로 변환된 필드
                                        'Duration': int(row['Duration']) if row['Duration'] else None,
                                        'Estimated_carbon_reduction': estimated_carbon_reduction,  # % 필드
                                        'Estimated_carbon_reduction_mt': estimated_carbon_reduction_mt,  # Million Tons 필드
                                        'Country_id': row['Country_id'],
                                        'Country': row['Country'],
                                    },
                                    ConditionExpression=Attr('Policy_id').not_exists()  # 중복 방지 조건
                                )
                                logger.info(f"Inserted item with Policy_id: {row['Policy_id']}")
                            except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                                # 아이템이 이미 존재할 경우, 아무 작업도 하지 않음
                                logger.warning(f"Item with Policy_id {row['Policy_id']} already exists.")
                            except Exception as e:
                                logger.error(f"Error inserting item with Policy_id {row['Policy_id']}: {e}")
                except Exception as e:
                    logger.error(f"Error reading CSV file {download_path}: {e}")
            else:
                logger.info(f"Ignored file: {object_key}")
    
    return {
        'statusCode': 200,
        'body': f'Data loaded into table {table_name} from {object_key}.'
    }
