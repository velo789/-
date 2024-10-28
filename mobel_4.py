import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import csv
import os

# 탄소 배출 모델 학습 및 저장 함수
def train_and_save_carbon_model(data_path, model_filename):
    # 데이터 로드
    data = pd.read_csv(data_path)

    # X (설명 변수)와 y (목표 변수) 설정
    X = data[['Population_total', 'GDP_current_USD (current US$)', 'GDP_growth_percent (annual %)']]
    y = data['carbon_emissions_value (MtCO2e GWP-AR4)']

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Carbon Model - Mean Squared Error: {mse}")
    print(f"Carbon Model - R-squared: {r2}")

    # 모델 저장
    joblib.dump(model, model_filename)
    print(f"탄소 배출 예측 모델이 {model_filename} 파일로 저장되었습니다.")

# 정책 데이터에 탄소 배출량 예측값 추가 함수
def add_predicted_emissions_to_policies(policies_data_path, combined_data_path, model_filename, output_path):
    # 저장된 모델 불러오기
    loaded_model = joblib.load(model_filename)

    # 정책 데이터 로드
    policies_data = pd.read_csv(policies_data_path)
    combined_data = pd.read_csv(combined_data_path)

    # 함수: 정책 데이터에서 나라 ID와 연도를 받아 탄소 데이터를 기반으로 예측값을 추가
    def predict_carbon_reduction(row):
        country_id = row['Country_id']
        year = row['Year']
        match = combined_data[(combined_data['Country_id'] == country_id) & (combined_data['Year'] == year)]
        if not match.empty:
            X = match[['Population_total', 'GDP_current_USD (current US$)', 'GDP_growth_percent (annual %)']].replace('N/A', 0).astype(float)
            y_pred_carbon = loaded_model.predict(X)
            return y_pred_carbon[0]
        else:
            return None

    # 정책 데이터에 예측값 추가
    policies_data['Predicted_Carbon_Emissions (MtCO2e)'] = policies_data.apply(predict_carbon_reduction, axis=1)
    policies_data_cleaned = policies_data.dropna(subset=['Predicted_Carbon_Emissions (MtCO2e)'])
    policies_data_filled = policies_data_cleaned.replace(r'^\s*$', 'N/A', regex=True).fillna('N/A')

    # 결과 저장
    policies_data_filled.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
    print(f"정책 데이터가 '{output_path}'로 저장되었습니다.")

# 정책 효과 모델 학습 및 저장 함수
def train_and_save_policy_effectiveness_model(data_path, model_filename):
    # 데이터 로드
    data = pd.read_csv(data_path)

    # X와 y 설정
    X = data[['Predicted_Carbon_Emissions (MtCO2e)', 'Implementation Cost', 'Duration']].replace('N/A', 0).astype(float)
    y = data['Estimated Carbon Reduction (Million Tons)'].replace('N/A', 0).astype(float)

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Policy Model - R-squared: {r_squared}")
    print(f"Policy Model - Mean Squared Error: {mse}")

    # 모델 저장
    joblib.dump(model, model_filename)
    print(f"정책 효과 모델이 '{model_filename}'로 저장되었습니다.")

# 메인 함수
if __name__ == "__main__":
    # 탄소 배출 모델 학습 및 저장
    train_and_save_carbon_model('final_combined_data_filtered.csv', 'carbon_emissions_model.pkl')
    
    # 정책 데이터에 탄소 배출량 예측값 추가
    add_predicted_emissions_to_policies('updated_policies_with_years_and_country_id.csv', 'final_combined_data_filtered.csv', 'carbon_emissions_model.pkl', 'updated_policies_with_predicted_carbon_reduction_filled.csv')
    
    # 정책 효과 모델 학습 및 저장
    train_and_save_policy_effectiveness_model('updated_policies_with_predicted_carbon_reduction_filled.csv', 'policy_effectiveness_model.pkl')
