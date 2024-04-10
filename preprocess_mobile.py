import pandas as pd

def preprocess_mobile(data):
    
    data['battery_power'] = data['battery_power'].apply(lambda x : 0 if x <= 1000
                                                        else (1 if 1000 < x <= 1500                                                     
                                                        else  2))

    data['clock_speed'] = data['clock_speed'].apply(lambda x : 0 if x <= 1 
                                                    else (1 if 1 < x <= 2                                                        
                                                    else  2))
    
    data['int_memory'] = data['int_memory'].apply(lambda x : 0 if x <= 10
                                                else (1 if 10 < x <= 20                                                      
                                                else (2 if 20 < x <= 30
                                                else (3 if 30 < x <= 40
                                                else (4 if 40 < x <= 50
                                                else (5 if 50 < x <= 60
                                                else  6))))))
    
    data['m_dep'] = data['m_dep'].apply(lambda x : 0 if x <= 0.5 
                                        else 1)
    
    data['mobile_wt'] = data['mobile_wt'].apply(lambda x : 0 if x <= 120
                                                else (1 if 120 < x <= 160                                                      
                                                else  2))

    data['n_cores'] = data['n_cores'].apply(lambda x : 0 if x <= 2
                                                else (1 if 2 < x <= 4                                                      
                                                else (2 if 4 < x <= 6                                                      
                                                else  3)))
    
    data['px_height'] = data['px_height'].apply(lambda x : 0 if x <= 720
                                                else (1 if 720 < x <= 1080                                                      
                                                else  2))
    
    data['px_width'] = data['px_width'].apply(lambda x : 0 if x <= 1080
                                              else (1 if 1080 < x <= 1440                                                      
                                              else  2))
    
    data['ram'] = data['ram'].apply(lambda x : 0 if x <= 1024
                                    else (1 if 1024 < x <= 2048                                                      
                                    else (2 if 2048 < x <= 3072                                                      
                                    else  3)))

    data['sc_h'] = data['sc_h'].apply(lambda x : 0 if x <= 12
                                    else 1)
    
    data['sc_w'] = data['sc_w'].apply(lambda x : 0 if x <= 7.2
                                    else 1)

    data['fc'] = data['fc'].apply(lambda x : 0 if x <= 3.8
                                  else (1 if 3.8 < x <= 7.6                                                      
                                  else  2))
    
    data['pc'] = data['pc'].apply(lambda x : 0 if x <= 10
                                  else 1)

    data['talk_time'] = data['talk_time'].apply(lambda x : 0 if x <= 11
                                                else 1)

    subgroup_dict = {subgroup: i for i, subgroup in enumerate(sorted(set(data['dual_sim'])))}
    data['dual_sim'] = data['dual_sim'].replace(subgroup_dict)


    feature_list = list(data.columns)
    feature_list.remove('dual_sim')
    feature_list.remove('price_range')

    feature_dict = {'price_range': 'y',
                    'dual_sim': 'subgroup'}

    for i, feat in enumerate(feature_list):
        # count_missing = data[feat].value_counts().get('?')
        # print(feat, ', num distinct =', len(set(data[feat])), ', missing tuples =', count_missing)
        feature_dict[feat] = f"f_{i}"

    data = data.rename(columns=feature_dict)
    data.reset_index(drop=True, inplace=True)

    return data


if __name__ == "__main__":
    data = pd.read_csv('./train.csv')
    df = preprocess_mobile(data)
    # print(data.head(100))

    df.to_csv('mobile_data_processed.csv', index=False)