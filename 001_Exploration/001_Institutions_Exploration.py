import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
import unidecode
import re
from random import sample

rutaDatos = '../Datos'
ruta ='./'

# Exploring the ROR Data to Create Artificial Training Data
# Data was downloaded from the ROR website for the date seen in the file string below
# -------> https://ror.readme.io/docs/data-dump
ror = pd.read_json(f"{rutaDatos}/v1.50-2024-07-29-ror-data.json")

ror = ror[ror.country.apply(lambda x: x['country_code'] == 'AR')]  #Me quedo sólo con las argentinas

print(f'Cantidad de entidades argentinas en archivo v1 {ror.shape}')
print('ror - Primeras 2: ')
print(ror.head(2))

#Agregamos nueva información a la lista
ror['alias_len'] = ror['aliases'].apply(len)
ror['acronyms_len'] = ror['acronyms'].apply(len)
ror['labels_len'] = ror['labels'].apply(len)
ror['address_len'] = ror['addresses'].apply(len)
ror['address'] = ror['addresses'].apply(lambda x: x[0])
ror['ror_id'] = ror['id'].apply(lambda x: x.split("/")[-1])
ror['types'] = ror['types'].apply(lambda x: x[0])

print('ror - primeras 2 con nuevas columnas de información')
print(ror.head(2))

# this file is not provided but the needed data is all institutions in OpenAlex
# with the following columns: 'ror_id','affiliation_id'
insts = pd.read_parquet(f"{rutaDatos}/OA_static_institutions_single_file.parquet",columns=['affiliation_id','ror_id'])
print(f'Cantidad entidades en OA_static_institutions_single_file.parque {insts.shape}')
print('insts - Primeras 2: ')
print(f'{insts.head(2)}')

ror_to_join = ror[['ror_id','name','aliases','acronyms','labels','country','types','address','alias_len','acronyms_len','labels_len','address_len']].copy()

def get_geoname_admin(address_dict):
    try:
        geoname_admin = address_dict['geonames_city']['geonames_admin1']['name']
    except:
        geoname_admin = "None"

    return geoname_admin

ror_to_join['country_code'] = ror_to_join['country'].apply(lambda x: x['country_code'])
ror_to_join['country_name'] = ror_to_join['country'].apply(lambda x: x['country_name'])
ror_to_join['city'] = ror_to_join['address'].apply(lambda x: x['city'])
ror_to_join['state'] = ror_to_join['address'].apply(lambda x: x['state'])
ror_to_join['region'] = ror_to_join['address'].apply(get_geoname_admin)
ror_to_join['institution'] = ror_to_join['name']

print('ror_to_join:')
print(ror_to_join.head(2))

codes_to_ignore = ['ja','fa','hi','ko','bn','zh','ml','ru','el','kn','gu','mk','ne','te','hy',\
                   'km','ti','kk','th','my','uk','pa','bg','ur','vi','ar','sr','he','ta','ka',\
                   'am','mr','lo','mn','be','or','ba','si','ky','uz']

labels = ror_to_join['labels'].explode().dropna().reset_index()
labels['label'] = labels['labels'].apply(lambda x: x['label'])
labels['iso639'] = labels['labels'].apply(lambda x: x['iso639'])

print('labels')
print(labels[~labels['iso639'].isin(codes_to_ignore)].sample(20))

# Getting string beginnings
# Looking to introduce more variety into the artificial strings so that they include some header information such as "College of .." or "Department of...".

with open(f"{ruta}/ror_string_beginnings/Company", 'r') as f:
    company_begs = f.readlines()

company_begs = list(set([x.rstrip('\n') for x in company_begs]))

with open(f"{ruta}/ror_string_beginnings/Education_dept", 'r') as f:
    education_dept_begs = f.readlines()

education_dept_begs = list(set([x.rstrip('\n') for x in education_dept_begs]))

with open(f"{ruta}/ror_string_beginnings/Education_college", 'r') as f:
    education_col_begs = f.readlines()

education_col_begs = list(set([x.rstrip('\n') for x in education_col_begs]))

with open(f"{ruta}/ror_string_beginnings/Education_school", 'r') as f:
    education_school_begs = f.readlines()

education_school_begs = list(set([x.rstrip('\n') for x in education_school_begs]))

with open(f"{ruta}/ror_string_beginnings/Healthcare", 'r') as f:
    healthcare_begs = f.readlines()

healthcare_begs = list(set([x.rstrip('\n') for x in healthcare_begs]))

all_education = []
for beg in education_col_begs:
    all_education.append(f"College of {beg}")

for beg in education_dept_begs:
    all_education.append(f"Department of {beg}")
    all_education.append(f"Dep of {beg}")
    all_education.append(f"Dept of {beg}")
    all_education.append(f"Dep. of {beg}")
    all_education.append(f"Dept. of {beg}")

for beg in education_school_begs:
    all_education.append(f"School of {beg}")
    all_education.append(f"Sch. of {beg}")
    all_education.append(f"Sch of {beg}")
len(all_education)

all_company = company_begs

all_healthcare = []
for beg in healthcare_begs:
    all_healthcare.append(f"Department of {beg}")
    all_healthcare.append(f"Dep of {beg}")
    all_healthcare.append(f"Dept of {beg}")
    all_healthcare.append(f"Dep. of {beg}")
    all_healthcare.append(f"Dept. of {beg}")

# Creating the artificial affiliation strings

#We would like to take advantage of the ROR data and use it to supplement/augment the current affiliation string data in OpenAlex. This could potentially allow for the Institution Tagger model to predict on institutions that have not yet had affiliation strings added to OpenAlex.
    
type_preinst_dict = {'Company': all_company,
                     'Education': all_education,
                     'Healthcare': all_healthcare}

def create_affiliation_string_from_scratch(institution, city, state, country, region, aliases, labels,
                                           acronyms, inst_type):
    aff_strings = []
    if aliases:
        aliases = [institution] + aliases
    else:
        aliases = [institution]
    if labels:
        for label in labels:
            if label['iso639'] in codes_to_ignore:
                pass
            else:
                aliases.append(label['label'])
    if acronyms:
        for acronym in acronyms:
            aliases.append(acronym)
    for alias in aliases:
        if "r&d" not in alias.lower():
            alias = alias.replace(" & ", " and ")
        match_string = unidecode.unidecode(alias)
        match_string = match_string.lower().replace(" ", "")
        match_string = "".join(re.findall("[a-zA-Z0-9]+", match_string))
        alias = unidecode.unidecode(alias)
        if ((state is None) & (region != city) & (city is not None) &
            (country is not None) & (region is not None)):
            region = unidecode.unidecode(region)
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {region}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {region}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias}, {region}", match_string])
            aff_strings.append([f"{alias} {city} {region} {country}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {city} {region}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias} {region}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(5):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(2):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {region}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {region}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])

        elif (state is not None) & (city is not None) & (country is not None):
            state = unidecode.unidecode(state)
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {state}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}, {state}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias}, {state}", match_string])
            aff_strings.append([f"{alias} {city} {state} {country}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {city} {state}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias} {state}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {state}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {state}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
        elif (city is not None) & (country is not None):
            country = unidecode.unidecode(country)
            city = unidecode.unidecode(city)
            aff_strings.append([f"{alias}, {city}, {country}", match_string])
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias}, {city}", match_string])
            aff_strings.append([f"{alias} {city} {country}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias} {city}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {city}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {city}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
        elif (country is not None):
            country = unidecode.unidecode(country)
            aff_strings.append([f"{alias}, {country}", match_string])
            aff_strings.append([f"{alias} {country}", match_string])
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(4):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])

                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                else:
                    for i in range(1):
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}, {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]} {country}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}, {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias} {country}", match_string])
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])

        else:
            aff_strings.append([f"{alias}", match_string])
            if inst_type in ['Education','Company','Healthcare']:
                list_to_sample = type_preinst_dict[inst_type]
                if inst_type == 'Education':
                    for i in range(5):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                elif inst_type == 'Healthcare':
                    for i in range(3):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
                else:
                    for i in range(2):
                        aff_strings.append([f"{sample(list_to_sample, 1)[0]} {alias}", match_string])
                        aff_strings.append([f"{alias} {sample(list_to_sample, 1)[0]}", match_string])
    return aff_strings

def get_institutions_list(institution, aliases, labels, acronyms):
    aff_strings = []
    if aliases:
        aliases = [institution] + aliases
    else:
        aliases = [institution]
    if labels:
        for label in labels:
            if label['iso639'] in codes_to_ignore:
                pass
            else:
                aliases.append(label['label'])
    return aliases

ror_to_join['affs'] = ror_to_join.apply(lambda x: get_institutions_list(x.institution, x.aliases,x.labels, x.acronyms), axis=1)
ror_to_join['aff_string'] = ror_to_join.apply(lambda x: create_affiliation_string_from_scratch(x.institution, x.city, x.state,x.country_name, x.region,x.aliases, x.labels, x.acronyms, x.types), axis=1)
ror_to_join['aff_string_len'] = ror_to_join['aff_string'].apply(len)
ror_to_join_final = ror_to_join.explode("aff_string").copy()

# Looking to quickly get combinations of city/region/country

art_empty_affs = ror_to_join[['city','region','country_name']].dropna().copy()

print('art_empty_affs: ')
print(art_empty_affs.shape)
print(art_empty_affs.sample(10))

art_empty_affs['original_affiliation_1'] = art_empty_affs.apply(lambda x: f"{x.city}, {x.country_name}", axis=1)

art_empty_affs['original_affiliation_2'] = art_empty_affs.apply(lambda x: f"{x.city}, {x.region}, {x.country_name}", axis=1)

city_country = art_empty_affs.sample(347).drop_duplicates()['original_affiliation_1'].to_list()

city_region_country = art_empty_affs.sample(347).drop_duplicates()['original_affiliation_2'].to_list()

#Some of these string will be used to train the model that only seeing a city/region/country should be a "no prediction"

artificial_empty_affs_DF = pd.DataFrame(zip(city_country+city_region_country),columns=['original_affiliation']).drop_duplicates(subset=['original_affiliation'])#\
#    .to_parquet(f"{ruta}/artificial_empty_affs.parquet")
artificial_empty_affs_DF.to_parquet(f"{ruta}/artificial_empty_affs.parquet")

print('artificial_empty_affs_DF: ')
print(artificial_empty_affs_DF.shape)
print(artificial_empty_affs_DF.head(1))

ror_to_join_final = ror_to_join[['ror_id','aff_string']].explode("aff_string").copy()
print('ror_to_join_final:')
print(ror_to_join_final.shape)
print(ror_to_join_final.head(1))

ror_to_join_final['original_affiliation'] = ror_to_join_final['aff_string'].apply(lambda x: x[0])
print('ror_to_join_final: ')
print(print(ror_to_join_final.head(2)))

ror_strings_Merge = ror_to_join_final.merge(insts, how='inner', on='ror_id')[['original_affiliation','affiliation_id']]

print('ror_strings_Merge:')
print(ror_strings_Merge.shape)
print(ror_strings_Merge.head(1))

print('Armando ror_strings.parquet')
ror_to_join_final.merge(insts, how='inner', on='ror_id')[['original_affiliation','affiliation_id']].to_parquet(f"{rutaDatos}/mag_and_ror_aff_strings_data/ror_strings.parquet")

print('FINALIZADO')
