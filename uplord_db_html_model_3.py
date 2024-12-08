'''
이 코드는 lambda 코드 입니다.
정책을 Dynamodb에 넣는 코드와 HTML에 나라 이름을 받아서 그 나라의 정책을 반환 해주고
정책을 클릭하였을 때 그 정책의 정보를 반환해주고 탄소 분석 모델과 정책 분석 모델을 통해
향후 탄소 배출량과 정책 적용시 향후 탄소 배출량을 예측할 수 있게 설계 하였습니다.
'''
import json
import boto3
import datetime
import io
import os
import csv
import logging
import pickle
from boto3.dynamodb.conditions import Attr
from decimal import Decimal
from botocore.exceptions import ClientError

# S3 컴그라인트 및 DynamoDB 리소스 초기화
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Policies')

# 로게 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# CloudWatch Logs에 출력 설정
log_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

def lambda_handler(event, context):
    logger.info("Lambda function has started execution.")
    # 이벤트 유형에 따라 처리 분기
    if 'Records' in event:  # S3 이벤트일 경우
        return handle_s3_event(event)
    else:  # API Gateway 이벤트일 경우
        return handle_api_request(event)

def handle_s3_event(event):
    # S3에서 CSV 파일 다운로드 및 DynamoDB 업데이트 처리
    for record in event['Records']:
        if record['eventName'].startswith('ObjectCreated:'):
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            
            # 특정 파일 이름 검색
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
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            try:
                                # 데이터 변환 및 DynamoDB 삽입
                                year = int(row['Year']) if row['Year'] else None
                                implementation_cost = int(Decimal(row['Implementation Cost'])) if row['Implementation Cost'] else None
                                estimated_carbon_reduction = Decimal(row['Estimated Carbon Reduction (%)']) if row['Estimated Carbon Reduction (%)'] else None
                                estimated_carbon_reduction_mt = Decimal(row['Estimated Carbon Reduction (Million Tons)']) if row['Estimated Carbon Reduction (Million Tons)'] else None
                                predicted_carbon_emissions = Decimal(row['Predicted_Carbon_Emissions (MtCO2e)']) if row['Predicted_Carbon_Emissions (MtCO2e)'] else None
                                
                                table.put_item(
                                    Item={
                                        'Policy_id': row['Policy_id'],
                                        'Policy_name': row['Policy'],
                                        'Policy_URL': row['Policy URL'],
                                        'Year': year,
                                        'Jurisdiction': row['Jurisdiction'],
                                        'Content': row['Content'],
                                        'Topics': row['Topics'],
                                        'Policy_Types': row['Policy Types'],
                                        'Sectors': row['Sectors'],
                                        'Implementation_cost': implementation_cost,
                                        'Duration': int(row['Duration']) if row['Duration'] else None,
                                        'Estimated_carbon_reduction': estimated_carbon_reduction,
                                        'Estimated_carbon_reduction_mt': estimated_carbon_reduction_mt,
                                        'Predicted_Carbon_Emissions_MtCO2e': predicted_carbon_emissions,
                                        'Country_id': row['Country_id'],
                                        'Country': row['Country'],
                                    },
                                    ConditionExpression=Attr('Policy_id').not_exists()
                                )
                                logger.info(f"Inserted item with Policy_id: {row['Policy_id']}")
                            except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                                logger.warning(f"Item with Policy_id {row['Policy_id']} already exists.")
                            except Exception as e:
                                logger.error(f"Error inserting item with Policy_id {row['Policy_id']}: {e}")
                except Exception as e:
                    logger.error(f"Error reading CSV file {download_path}: {e}")
            else:
                logger.info(f"Ignored file: {object_key}")
    return {
        'statusCode': 200,
        'body': f'Data loaded into table from {object_key}.'
    }

def handle_api_request(event):
    logger.info(f"Received event: {event}")
    # API Gateway 요청 처리 (정책 목록 반환 및 예측 모델 실행)
    path = event['requestContext']['resourcePath']
    method = event['httpMethod']

    if method == 'OPTIONS':
        logger.info("Handling OPTIONS request for CORS preflight.")
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps('CORS preflight request handled')
        }
    logger.info(f"Handling request for path: {path} with method: {method}")

    country_name = event.get('queryStringParameters', {}).get('country')
    selected_policy = event.get('queryStringParameters', {}).get('policy')

    # 모델 파일 및 S3 버킷 관련 정보
    bucket_name = 'your-s3-bucket-name'
    data_key = 'final_combined_data_filtered.csv'
    carbon_model_key = 'carbon_emissions_model.pkl'
    policy_model_key = 'policy_effectiveness_model.pkl'

    # 한국 데이터 필터링을 위한 처리
    if country_name and country_name.lower() in ['korea', 'south korea']:
        country_name = 'Korea, Rep.'

    try:
        # /project1/Policies 경로일 경우: 정책 목록 반환
        if path == '/project1/Policies' and method == 'GET':
            if not country_name:
                return {
                    'statusCode': 400,
                    'body': json.dumps('Country parameter is required')
                }

            # DynamoDB에서 해당 국가의 정책 목록 가져오기
            response = table.scan(
                FilterExpression=Attr('Country').eq(country_name)
            )

            if 'Items' not in response or not response['Items']:
                return {
                    'statusCode': 404,
                    'body': json.dumps(f'No policies found for country: {country_name}')
                }

            # 정책 목록과 상세 내용 추가
            policies = []

            for item in response['Items']:
                policies.append({
                    'Policy_name': item['Policy_name', None],
                    'Policy_id': item['Policy_id'],
                    'Country_id': item['Country_id'],
                    'Estimated_carbon_reduction': item.get('Estimated_carbon_reduction', None),
                    'Content': item.get('Content', ''),
                    'Policy_URL': item.get('Policy_URL', None)
                })

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({'policies': policies})
            }
        #정책을 선택한 후
        elif path == '/project1/predictions' and method == 'GET':
            if not country_name or not selected_policy:
                return {
                    'statusCode': 400,
                    'body': json.dumps('Country and policy parameters are required')
                }

            # S3에서 모델 파일 다운로드 및 로드
            carbon_model_path = f'/tmp/{carbon_model_key}'
            policy_model_path = f'/tmp/{policy_model_key}'

            s3.download_file(bucket_name, carbon_model_key, carbon_model_path)
            s3.download_file(bucket_name, policy_model_key, policy_model_path)

            with open(carbon_model_path, 'rb') as model_file:
                carbon_model = pickle.load(model_file)

            with open(policy_model_path, 'rb') as model_file:
                policy_model = pickle.load(model_file)

            # S3에서 CSV 데이터 파일 다운로드
            response = s3.get_object(Bucket=bucket_name, Key=data_key)
            data_body = response['Body'].read().decode('utf-8')

            # CSV 데이터 파싱
            csv_reader = csv.DictReader(io.StringIO(data_body))
            data = [row for row in csv_reader]

            # 선택된 Country_id를 기반으로 필터링
            year_to_data_temp = defaultdict(list)
            for row in data:
                if row['Country_id'] == country_id:
                    year_to_data_temp[int(row['Year'])].append({
                        "Population_total": float(row['Population_total']),
                        "GDP_current_USD": float(row['GDP_current_USD']),
                        "GDP_growth_percent": float(row['GDP_growth_percent']),
                    })

            # 연도별로 첫 번째 데이터를 선택
            year_to_data = {
                year: entries[0] for year, entries in year_to_data_temp.items()
            }

            # 예측 결과 저장
            predictions = []
            chart_data_carbon_emissions = []
            chart_data_policy_effect = []
            last_population = year_to_data.get(max(year_to_data.keys(), default=2016), {}).get("Population_total", 0)
            last_gdp = year_to_data.get(max(year_to_data.keys(), default=2016), {}).get("GDP_current_USD", 0)
            gdp_growth_percent = year_to_data.get(max(year_to_data.keys(), default=2016), {}).get("GDP_growth_percent", 0)

            # 정책 데이터에서 implementation_cost와 duration 추출
            try:
                policy_data = table.get_item(Key={'Policy_id': selected_policy})
                if 'Item' not in policy_data:
                    return {
                        'statusCode': 404,
                        'body': json.dumps(f'No policy found for ID: {selected_policy}')
                    }

                implementation_cost = policy_data['Item'].get('Implementation_cost', 0)
                duration = policy_data['Item'].get('Duration', 1)  # 기본값으로 1년을 가정
            except Exception as e:
                logger.error(f"Error retrieving policy data: {e}")
                return {
                    'statusCode': 500,
                    'body': json.dumps('Error retrieving policy data')
                }

            for year in range(2017, 2051):
                if year in year_to_data:
                    # 파일에 있는 데이터를 사용
                    population_total = year_to_data[year]["Population_total"]
                    gdp_current_usd = year_to_data[year]["GDP_current_USD"]
                    gdp_growth_percent = year_to_data[year]["GDP_growth_percent"]
                else:
                    # 파일에 데이터가 없는 경우 가상 데이터 생성
                    population_total = last_population * 1.01  # 인구 1% 증가 가정
                    gdp_current_usd = last_gdp * (1 + gdp_growth_percent / 100)  # GDP 성장 반영
                    last_population = population_total
                    last_gdp = gdp_current_usd

                # 탄소 배출량 예측
                carbon_input_data = [[population_total, gdp_current_usd, gdp_growth_percent]]
                predicted_carbon_emissions = carbon_model.predict(carbon_input_data)[0]
                chart_data_carbon_emissions.append(predicted_carbon_emissions)

                # 정책 효과 예측
                policy_input_data = [[predicted_carbon_emissions, implementation_cost, duration]]
                predicted_carbon_reduction = policy_model.predict(policy_input_data)[0]
                final_emissions_after_policy = predicted_carbon_emissions - predicted_carbon_reduction
                chart_data_policy_effect.append(final_emissions_after_policy)

                # 결과 저장
                predictions.append({
                    "Year": year,
                    "Predicted_Carbon_Emissions": predicted_carbon_emissions,
                    "Adjusted_Carbon_Emissions": final_emissions_after_policy
                })

            # 결과 반환
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': json.dumps({
                    'predictions': predictions,
                    'chart_data': {
                        'carbon_emissions': chart_data_carbon_emissions,
                        'policy_effect': chart_data_policy_effect
                    }
                })
            }
        else:
            logger.error(f"Invalid path: {path}")
            return {
                'statusCode': 404,
                'body': json.dumps('Invalid path')
            }

    except ClientError as e:
        logger.error(f"ClientError: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'error': str(e)})
        }
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps({'error': str(e)})
        }
