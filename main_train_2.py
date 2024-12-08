'''
이 코드는 수집된 정책 데이터와 탄소 분석 모델 데이터를 정제하는 코드입니다.
'''
import pandas as pd
import csv

# 파일에서 인구 데이터를 로드하고 클린업하는 함수
def load_and_clean_population_data(filepath):
    population_data = pd.read_csv(filepath)
    # 불필요한 열 제거
    if 'Unnamed: 0' in population_data.columns:
        population_data = population_data.drop(columns=['Unnamed: 0'])
    if '0' in population_data.columns:
        population_data = population_data.drop(columns=['0'])
    # 고유 국가 이름에 ID를 매핑
    unique_countries = population_data['Country Name'].unique()
    new_country_mapping = {country: i + 1 for i, country in enumerate(unique_countries)}
    population_data['Country_id'] = population_data['Country Name'].map(new_country_mapping)
    population_data['Country_id'] = population_data['Country_id'].astype(int)
    return population_data

# 인구 데이터를 피벗 해제(melt)하여 정렬하는 함수
def melt_population_data(population_data):
    population_melted = pd.melt(population_data, id_vars=['Country Name', 'Country Code', 'Country_id'],
                                value_vars=population_data.columns[4:],
                                var_name='year', value_name='Population_total')
    population_sorted = population_melted.sort_values(by=['Country_id', 'year'])
    return population_sorted

# 데이터를 파일로 저장하는 함수
def save_data(data, filepath):
    data.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

# GDP 데이터를 로드, 피벗 해제, 정렬하는 함수
def load_and_clean_gdp_data(filepath, value_name):
    gdp_data = pd.read_csv(filepath)
    gdp_data_melted = pd.melt(gdp_data, id_vars=['Country Name', 'Country Code', 'Indicator Name'],
                              value_vars=gdp_data.columns[4:],
                              var_name='year', value_name=value_name)
    gdp_sorted = gdp_data_melted.sort_values(by=['Country Name', 'Country Code', 'year'])
    return gdp_sorted

# 배출 데이터를 로드 및 클린업하는 함수
def load_and_clean_emissions_data(filepath):
    data = pd.read_csv(filepath)
    data_filtered = data[data['sector'] != 'LULUCF']
    data_sorted = data_filtered.sort_values(by=['region', 'year'])
    return data_sorted

# 지역 데이터를 인구 데이터와 매핑하는 함수
def map_region_data(region_data, population_data):
    population_data_unique = population_data.groupby('Country Name').first().reset_index()
    region_mapping = population_data_unique.set_index('Country Code')[['Country Name', 'Country_id']].to_dict(orient='index')

    def map_region(row):
        region_key = 'EUU' if row['region'] == 'EU27' else row['region']
        if region_key in region_mapping:
            return pd.Series([region_mapping[region_key]['Country Name'], region_key, region_mapping[region_key]['Country_id']])
        else:
            return pd.Series([None, row['region'], None])

    region_data[['Country Name', 'Country Code', 'Country_id']] = region_data.apply(map_region, axis=1)
    cols = ['Country Name', 'Country Code', 'Country_id'] + [col for col in region_data.columns if col not in ['Country Name', 'Country Code', 'Country_id', 'region']]
    region_data = region_data[cols]
    region_data.rename(columns={'value': 'carbon_emissions_value (MtCO2e GWP-AR4)'}, inplace=True)
    if 'unit' in region_data.columns:
        region_data = region_data.drop(columns=['unit'])
    return region_data

def get_eu27_value(row, data, column_name, eu27_country_ids, euu_country_id):
    country_id = row['Country_id']
    if country_id in eu27_country_ids:
        country_id = euu_country_id
    value = data[(data['Country_id'] == country_id) & (data['year'] == row['year'])]
    if not value.empty:
        return value[column_name].tolist()[0] if len(value[column_name].tolist()) > 0 else None
    else:
        return None

# 특정 열 값을 여러 행으로 확장하는 함수
def expand_rows_with_multiple_values(data, column_names):
    expanded_data = []
    for idx, row in data.iterrows():
        max_len = max([len(row[col]) if isinstance(row[col], list) else 1 for col in column_names])
        for i in range(max_len):
            new_row = row.copy()
            for col in column_names:
                if isinstance(row[col], list):
                    new_row[col] = row[col][i] if i < len(row[col]) else None
            expanded_data.append(new_row)
    return pd.DataFrame(expanded_data)

# 정책 데이터를 로드하고 클린업하는 함수
def load_and_clean_policies(filepath, population_data):
    policies_data = pd.read_csv(filepath)
    # 국가 이름 매핑
    country_name_mapping = {
        'Plurinational State of Bolivia': 'Bolivia',
        'Bolivarian Republic of Venezuela': 'Venezuela, RB',
        'Republic of Moldova': 'Moldova',
        'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
        "Cote D'ivoire": "Cote d'Ivoire",
        'Czech Republic': 'Czechia',
        'Vatican City': 'Holy See',
        'Saint Lucia': 'St. Lucia',
        'United Republic of Tanzania': 'Tanzania',
        'Micronesia (Federated States of)': 'Micronesia, Fed. Sts.',
        "People's Republic of China": 'China',
        'Chinese Taipei': 'Taiwan',
        'Islamic Republic of Iran': 'Iran, Islamic Rep.',
        "Lao People's Democratic Republic": 'Lao PDR',
        'Kyrgyzstan': 'Kyrgyz Republic',
        'Republic of the Congo': 'Congo, Rep.',
        'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
        'Korea': 'Korea, Rep.',
        'Egypt': 'Egypt, Arab Rep.',
        'Yemen': 'Yemen, Rep.'
    }
    reverse_mapping = population_data.set_index('Country Name')['Country_id'].to_dict()
    policies_data['Country'] = policies_data['Country'].replace(country_name_mapping)
    policies_data['Country_id'] = policies_data['Country'].map(reverse_mapping)
    current_max_id = max(reverse_mapping.values()) if reverse_mapping else 0
    for index, row in policies_data[policies_data['Country_id'].isna()].iterrows():
        if row['Country'] not in reverse_mapping:
            current_max_id += 1
            reverse_mapping[row['Country']] = current_max_id
            policies_data.at[index, 'Country_id'] = current_max_id
        else:
            policies_data.at[index, 'Country_id'] = reverse_mapping[row['Country']]
    policies_data['Country_id'] = policies_data['Country_id'].astype(int)
    # 빈 년도 값 채우기
    years_to_fill = list(range(1948, 1930, -1))
    year_index = 0
    for index, row in policies_data.iterrows():
        if pd.isnull(row['Year']) or row['Year'] == 'N/A':
            if year_index < len(years_to_fill):
                policies_data.at[index, 'Year'] = years_to_fill[year_index]
                year_index += 1
            else:
                break
    policies_data['Year'] = pd.to_numeric(policies_data['Year'], errors='coerce').astype('Int64')
    policies_data = policies_data.sort_values(by='Year', ascending=True).reset_index(drop=True)
    policies_data['Policy_id'] = policies_data.groupby('Country_id').cumcount() + 1
    policies_data.fillna('N/A', inplace=True)
    if 'Status' in policies_data.columns:
        policies_data.drop(columns=['Status'], inplace=True)
    return policies_data

def main():

    # Population data processing
    population_data = load_and_clean_population_data('Cleaned_Population_Data.csv')
    population_sorted = melt_population_data(population_data)
    save_data(population_sorted, 'sorted_population_data.csv')

    # GDP data processing (previous growth)
    gdp_previous_sorted = load_and_clean_gdp_data('Cleaned_gdp_data1.csv', 'previous_GDP_growth_percent')
    save_data(gdp_previous_sorted, 'sorted_gdp_growth_data_previous.csv')

    # GDP data processing (current USD)
    gdp_current_sorted = load_and_clean_gdp_data('cleaned_gdp_data_2.csv', 'GDP_current_USD')
    save_data(gdp_current_sorted, 'sorted_gdp_current_usd.csv')

    # Emissions data processing
    emissions_data_sorted = load_and_clean_emissions_data('08032024_CountryAssessmentData_no_capita-fixed_2.csv')
    save_data(emissions_data_sorted, 'sorted_data_without_LULUCF.csv')

    # Emissions data processing step 1
    region_data = pd.read_csv('sorted_data_without_LULUCF.csv')
    region_data = map_region_data(region_data, population_data)
    save_data(region_data, 'updated_region_data.csv')

    # Emissions data processing step 2
    population_data = pd.read_csv('sorted_population_data.csv')
    gdp_data_current = pd.read_csv('sorted_gdp_current_usd.csv')
    gdp_data_previous = pd.read_csv('sorted_gdp_growth_data_previous.csv')
    carbon_data = pd.read_csv('updated_region_data.csv')

    # 새로운 GDP 데이터 정제 코드 추가
    # CSV 파일 로드
    population_data = pd.read_csv('sorted_population_data.csv')
    gdp_data_current = pd.read_csv('sorted_gdp_current_usd.csv')
    gdp_data_previous = pd.read_csv('sorted_gdp_growth_data_previous.csv')

    # 중복되지 않는 Country ID를 가져오기 위한 함수
    def assign_country_id(row, country_id_mapping):
        if row['Country Name'] in country_id_mapping:
            return country_id_mapping[row['Country Name']]
        return None

    # Population 데이터에서 Country Name과 Country ID를 매핑
    country_id_mapping = population_data.set_index('Country Name')['Country_id'].to_dict()

    # GDP 데이터에 Country ID 추가
    gdp_data_current['Country_id'] = gdp_data_current.apply(assign_country_id, axis=1, country_id_mapping=country_id_mapping)
    gdp_data_previous['Country_id'] = gdp_data_previous.apply(assign_country_id, axis=1, country_id_mapping=country_id_mapping)

    # Country ID를 정수형으로 변환
    gdp_data_current['Country_id'] = gdp_data_current['Country_id'].astype(int, errors='ignore')
    gdp_data_previous['Country_id'] = gdp_data_previous['Country_id'].astype(int, errors='ignore')

    # 열 이름에 단위 추가
    gdp_data_current.rename(columns={'GDP_current_USD': 'GDP_current_USD (current US$)'}, inplace=True)
    gdp_data_previous.rename(columns={'previous_GDP_growth_percent': 'GDP_growth_percent (annual %)'}, inplace=True)

    # 'Indicator Name' 열을 삭제
    gdp_data_current.drop(columns=['Indicator Name'], inplace=True)
    gdp_data_previous.drop(columns=['Indicator Name'], inplace=True)

    # 업데이트된 데이터를 CSV로 저장, quoting과 quotechar 옵션 추가
    gdp_data_current.to_csv('updated_gdp_data_current_with_units.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')
    gdp_data_previous.to_csv('updated_gdp_data_previous_with_units.csv', index=False, quoting=csv.QUOTE_NONNUMERIC, quotechar='"')

    # Emissions data processing
    emissions_data_sorted = load_and_clean_emissions_data('08032024_CountryAssessmentData_no_capita-fixed_2.csv')
    save_data(emissions_data_sorted, 'sorted_data_without_LULUCF.csv')

    # Emissions data processing step 1
    region_data = pd.read_csv('sorted_data_without_LULUCF.csv')
    region_data = map_region_data(region_data, population_data)
    save_data(region_data, 'updated_region_data.csv')

    # Emissions data processing step 2
    population_data = pd.read_csv('sorted_population_data.csv')
    gdp_data_current = pd.read_csv('updated_gdp_data_current_with_units.csv')
    gdp_data_previous = pd.read_csv('updated_gdp_data_previous_with_units.csv')
    carbon_data = pd.read_csv('updated_region_data.csv')

    eu27_country_ids = [15, 18, 22, 54, 55, 59, 71, 72, 76, 79, 90, 100, 102, 111, 116, 143, 144, 145, 159, 176, 194, 200, 221, 222]
    euu_country_id = population_data.loc[population_data['Country Code'] == 'EUU', 'Country_id'].values[0]

    population_data['GDP_current_USD (current US$)'] = population_data.apply(get_eu27_value, axis=1, data=gdp_data_current, column_name='GDP_current_USD (current US$)', eu27_country_ids=eu27_country_ids, euu_country_id=euu_country_id)
    population_data['GDP_growth_percent (annual %)'] = population_data.apply(get_eu27_value, axis=1, data=gdp_data_previous, column_name='GDP_growth_percent (annual %)', eu27_country_ids=eu27_country_ids, euu_country_id=euu_country_id)
    population_data['carbon_emissions_value (MtCO2e GWP-AR4)'] = population_data.apply(get_eu27_value, axis=1, data=carbon_data, column_name='carbon_emissions_value (MtCO2e GWP-AR4)', eu27_country_ids=eu27_country_ids, euu_country_id=euu_country_id)

    column_names_to_expand = ['GDP_current_USD (current US$)', 'GDP_growth_percent (annual %)', 'carbon_emissions_value (MtCO2e GWP-AR4)']
    population_data_expanded = expand_rows_with_multiple_values(population_data, column_names_to_expand)

    mask = population_data_expanded['carbon_emissions_value (MtCO2e GWP-AR4)'].notnull()
    population_data_expanded.loc[mask, 'GDP_current_USD (current US$)'] = population_data_expanded.loc[mask, 'GDP_current_USD (current US$)'].fillna(method='ffill')
    population_data_expanded.loc[mask, 'GDP_growth_percent (annual %)'] = population_data_expanded.loc[mask, 'GDP_growth_percent (annual %)'].fillna(method='ffill')
    population_data_expanded.rename(columns={'year': 'Year'}, inplace=True)
 
    data_cleaned = population_data_expanded.dropna(subset=['Population_total', 'GDP_current_USD (current US$)',
                                                           'GDP_growth_percent (annual %)', 'carbon_emissions_value (MtCO2e GWP-AR4)'])
    min_year = data_cleaned['Year'].min()
    max_year = data_cleaned['Year'].max()
    data_filtered = data_cleaned[(data_cleaned['Year'] >= min_year) & (data_cleaned['Year'] <= max_year)]
    save_data(data_filtered, 'final_combined_data_filtered.csv')

    # Policies data processing
    policies_data = load_and_clean_policies('new_policy.csv', population_data)
    save_data(policies_data, 'updated_policies_with_years_and_country_id.csv')

if __name__ == "__main__":
    main()
