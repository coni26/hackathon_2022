import pandas as pd
from tmdbv3api import TMDb
from tmdbv3api import Movie
tmdb = TMDb()
tmdb.api_key = '7cef6d8c76b28ffeab3b87622e41d61e'
from tqdm.auto import tqdm
tqdm.pandas()
from ast import literal_eval
import numpy as np
from transformers import pipeline
import pickle

df = pd.read_json('tmdb_data.json')

def predict_bechdel(df):
    df = df[df["release_date"] != ""]
    df.drop_duplicates('imdb_id', inplace=True)
    df["year"] = df["release_date"].apply(lambda x: int(x[:4]))
    df.dropna(subset=["runtime"],inplace=True)
    df["runtime"] = df["runtime"].apply(int)
    
    df["n_actors"], df["n_women"],df["is_realisator_woman"], df["is_writer_woman"] = zip(*df['id'].map(get_stats_genders))

    df["production_countries"] = df["production_countries"].apply(clean_country)

    top10_countries = ['United States of America','United Kingdom','France','Germany','Canada','Japan','Italy','Spain','Australia','Sweden']
    
    df["production_countries"] = df["production_countries"].apply(lambda x: synthetize_countries(x,top10_countries))
    
    df["production_companies"] = df["production_companies"].apply(clean_prod)
    
    top10_companies = ['Warner Bros. Pictures','Universal Pictures','Paramount','20th Century Fox','Columbia Pictures','Metro-Goldwyn-Mayer','Canal+','Walt Disney Pictures','New Line Cinema','United Artists']
    
    df["production_companies"] = df["production_companies"].apply(lambda x: synthetize_companies(x,top10_companies))
    
    df["keywords"] = df["id"].apply(get_keywords)
    
    df["pegi"]= df["id"].apply(get_pegi)
    
    df["pegi"] = df["pegi"].replace({'10' : '12',
                              '11' : '12',
                              '13' : '12',
                               '14': '12',
                              '15' : '16',
                              '17':'16',
                              None : '0'})
    
    df['pegi'] = 'pegi' + df['pegi'].astype(str)
    
    df = df.join(pd.get_dummies(df['pegi']))
    
    df["is_netflix"], df["is_prime"] = zip(*df["id"].progress_map(is_netflix_prime)) 
    
    df["genres"] = df["genres"].apply(lambda l: [i.get("name") for i in l])
    
    for col in ["production_companies", "production_countries", "genres"]:
        df = multiple_dummies(df, col)
    
    classifier = pipeline("zero-shot-classification")
    df["score_woman"] = df["keywords"].apply(lambda x: theme(x, ['woman', 'girl'], classifier))
    df['score_woman'] = df['score_woman'].fillna(0.34)
    
    for colonne in ['budget', 'year', 'n_actors', 'n_women', 'is_realisator_woman', 'is_writer_woman', 'pegi0', 'pegi12', 'pegi16', 'pegi18', 'is_netflix',
       'is_prime', '20th Century Fox', 'Canal+', 'Columbia Pictures',
       'Metro-Goldwyn-Mayer', 'New Line Cinema', 'Other_company', 'Paramount',
       'United Artists', 'Universal Pictures', 'Walt Disney Pictures',
       'Warner Bros. Pictures', 'Australia', 'Canada', 'France', 'Germany',
       'Italy', 'Japan', 'Other_country', 'Spain', 'Sweden', 'United Kingdom',
       'United States of America', 'Action', 'Adventure', 'Animation',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
       'TV Movie', 'Thriller', 'War', 'Western', 'score_woman','runtime']:
        if not colonne in df.columns:
            df[colonne] = 0
    

    X = df[['budget', 'year', 'n_actors', 'n_women', 'is_realisator_woman', 'is_writer_woman', 'pegi0', 'pegi12', 'pegi16', 'pegi18', 'is_netflix',
       'is_prime', '20th Century Fox', 'Canal+', 'Columbia Pictures',
       'Metro-Goldwyn-Mayer', 'New Line Cinema', 'Other_company', 'Paramount',
       'United Artists', 'Universal Pictures', 'Walt Disney Pictures',
       'Warner Bros. Pictures', 'Australia', 'Canada', 'France', 'Germany',
       'Italy', 'Japan', 'Other_country', 'Spain', 'Sweden', 'United Kingdom',
       'United States of America', 'Action', 'Adventure', 'Animation',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
       'TV Movie', 'Thriller', 'War', 'Western', 'score_woman','runtime']]
    
    
    clf = pickle.load(open('finalized_model.sav', 'rb'))
    
    return clf.predict(X)
    
def get_stats_genders(id):
    '''
        Récupération du genre des acteurs via l'api
    '''
    try:
        movie = Movie()
        m = movie.details(id)
        n_actors=0
        n_women=0
        for actor in m['casts']['cast']:
            if actor.get('known_for_department') == 'Acting':
                n_actors+=1
            if (actor.get('known_for_department') == 'Acting') & (actor.get('gender')==1):
                n_women+=1
        is_realisator_woman = 0
        is_writer_woman = 0
        for crew in m['casts']['crew']:
            if (crew.get('job')=='Director') & (crew.get('gender')==1):
                is_realisator_woman = 1
            if (crew.get('job')=='Writer') & (crew.get('gender')==1):
                is_writer_woman = 1 
        return([n_actors,n_women,is_realisator_woman,is_writer_woman])
    except:
        return([0,0,0,0])    
    
    

def clean_country(l):
    '''
        Garde uniquement le nom des pays depuis un dictionnaire
    '''
    countries = []
    if len(l) > 0:
        for country in l:
            countries.append(country.get('name'))
    return(countries)

def synthetize_countries(l,top):
    '''
        Garde uniquement les 10 principaux pays
    '''
    if len(l)>0:
        for i,x in enumerate(l):
            if x not in top:
                l[i]="Other_country"
    return(list(set(l)))

def clean_prod(l):
    '''
        Garde uniquement le nom des entreprises de production depuis un dictionnaire
    '''
    prod=[]
    if len(l)>0:
        for cies in l:
            prod.append(cies.get('name'))
    return(prod)

def synthetize_companies(l,top):
    '''
        Garde uniquement les 10 principales entreprises de production
    '''
    if len(l)>0:
        for i,x in enumerate(l):
            if x not in top:
                l[i]="Other_company"
    return(list(set(l)))

def get_keywords(id):
    '''
        Récupère les keywords des films depuis l'api
    '''
    try:
        ll=[]
        movie = Movie()
        m = movie.details(id)
        for kw in m["keywords"]["keywords"]:
            ll.append(kw.get("name")) 
        return(ll)
    except:
        return([])
    
def get_pegi(id):
    '''
        Récupère les certification pegi
    '''
    try:
        movie = Movie()
        m = movie.details(id)
        for dic in m["release_dates"].get('results'):
            l = dic.get("release_dates")
            if ("1" in l[0].get("certification")) & (len(l[0].get("certification")) == 2):
                return(l[0].get("certification"))
    except:
        return(None)


def is_netflix_prime(id):
    '''
        Récupère la plateforme netflix ou prime
    '''
    try:
        l=[0,0]
        movie = Movie()
        m = movie.details(id)
        for dic in m["release_dates"].get('results'):
            rel = dic.get("release_dates")
            for x in rel:
                if x.get("note") == "Netflix":
                    l[0]=1
                if x.get("note") == "Prime Video":
                    l[1]=1
        return(l)
    except:
        return([0,0])
    
def multiple_dummies(df, column):
    '''
        Fonction générique pour one hot encode
    '''
    df_bis = pd.DataFrame(
{
    'imdb_id' : df['imdb_id'].values.repeat(df[column].str.len(), axis=0),
    column : np.concatenate(df[column].tolist())
})
    df_bis = df_bis.set_index('imdb_id')[column].str.get_dummies().sum(level=0)
    cols = df_bis.columns
    df = pd.merge(df, df_bis, on = 'imdb_id', how = "left")
    df[cols] = df[cols].fillna(0.0).astype(int)
    return(df)

def theme(l, words, classifier):
    '''
        Donne le score la classification des mots de l dans words
    '''
    if len(l)>0:
        for x in l:
            return classifier(x, words, multi_label=True).get("scores")[0]
