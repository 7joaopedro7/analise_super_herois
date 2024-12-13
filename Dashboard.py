import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import seaborn as sns

import joblib
from sklearn.ensemble import RandomForestClassifier

skills = ['Physical Skills', 'Energy Manipulation', 'Elemental Powers', 'Mental and Psychic Powers', 'Space-Time Manipulation',
        'Body Transformation', 'Supernatural Forces', 'Tech Skills', 'Perception Skills', 'Locomotion Skills', 'Combat Skills']

races = ['Race_group_by_Alien', 'Race_group_by_Artificial Beings', 'Race_group_by_Divine and Semidivine', 'Race_group_by_Genetically Altered Beings',
    'Race_group_by_Human and Variation', 'Race_group_by_Mythical', 'Race_group_by_Not Identified', 'Race_group_by_Underwater']

classificacao_realizada = 'Ainda não foi realizada nenhuma classificação'

def radar_chart(categories, values, title = "Radar Chart"):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize = (5, 5), subplot_kw = dict(polar = True))

    ax.fill(angles, values, color = 'blue', alpha = 0.25)
    ax.plot(angles, values, color = 'blue', linewidth = 2)

    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize = 10)

    ax.set_title(title, fontsize = 16, pad = 20)

    return fig
    
def analise_de_cluster(data, Cluster, grafico):
    
    cluster_data = data[data.Cluster == Cluster]

    if grafico == 'skill':
        skill_media = list()
        for skill in skills:
            skill_media.append(round(cluster_data[skill].mean(), 2))

        return radar_chart(skills, skill_media, f"Distribuição de poderes no cluster {Cluster}")

    if grafico == 'raca':
        raca_media = list()
        for race in races:
            raca_media.append(round(cluster_data[race].sum(), 2))

        return radar_chart(races, raca_media, f"Distribuição de raças no cluster {Cluster}")

    if grafico == 'genero':

        fig, ax = plt.subplots(figsize = (3.36, 3.36))

        cluster_data['Gender'].replace('-', 'Others / Not Identified', inplace = True)

        values = cluster_data['Gender'].value_counts().values
        labels = cluster_data['Gender'].value_counts().index
        total = cluster_data['Gender'].value_counts().sum()

        plt.pie(values, labels = labels, autopct=lambda p: '{:.0f}'.format(p * total / 100))
        plt.title(f'Distribuição de gêneros no Cluster {Cluster}')

        return fig

    if grafico == 'alinhamento':

        fig, ax = plt.subplots(figsize = (3.36, 3.36))

        cluster_data['Alignment'].replace('-', 'Others / Not Identified', inplace = True)

        values = cluster_data['Alignment'].value_counts().values
        labels = cluster_data['Alignment'].value_counts().index
        total = cluster_data['Alignment'].value_counts().sum()

        plt.pie(values, labels = labels, autopct=lambda p: '{:.0f}'.format(p * total / 100))
        plt.title(f'Distribuição de alinhamentos no Cluster {Cluster}')

        return fig

def grafico_raca(df_grafico, hue_alinhamento):

    fig, ax = plt.subplots(figsize = (6, 3))
    sns.set_style('whitegrid')

    if hue_alinhamento:

        df_raca = df_grafico[['Race_group_by', 'Gender', 'Publisher_group_by', 'Alignment']].\
            groupby(['Race_group_by', 'Alignment']).count()[['Gender']].reset_index().\
                rename(columns = {'Gender': 'Count'})

        sns.barplot(data = df_raca, x = 'Race_group_by', y = 'Count', hue = 'Alignment', ax = ax, color = 'lightgreen')

    else:

        df_raca = df_grafico[['Race_group_by', 'Gender', 'Publisher_group_by', 'Alignment']].\
            groupby('Race_group_by').count()[['Gender']].reset_index().\
                rename(columns = {'Gender': 'Count'})

        sns.barplot(data = df_raca, x = 'Race_group_by', y = 'Count', ax = ax, color = 'lightgreen')

    plt.title('Distribuição de raças')
    plt.xticks(rotation = 90)

    return fig

def grafico_skill(df_grafico, raca):

    fig, ax = plt.subplots(figsize = (6, 3))
    sns.set_style('whitegrid')
  
    df_skills = df_grafico[skills + ['Race_group_by']]
    df_skills = df_skills[df_skills.Race_group_by == raca].drop(columns = 'Race_group_by')

    sns.barplot(x = df_skills.mean().index, y = df_skills.mean().values, color = 'lightgreen', ax = ax)
    plt.title('Distribuição média de poderes por raça')
    plt.xticks(rotation = 90)

    return fig

def classificar_personagem(model, habilidades, caracteristicas):
    
    # editora - genero - raça

    alinhamento = None

    dados_personagem = pd.DataFrame({})

    # Habilidades
    for i, skill in enumerate(skills):
        dados_personagem[skill] = [habilidades[i]]

    # Genero
    if caracteristicas[3] == 'Feminino':
        dados_personagem['Gender_Female'] = [1]
    else:
        dados_personagem['Gender_Female'] = [0]

    if caracteristicas[3] == 'Masculino':
        dados_personagem['Gender_Male'] = [1]
    else:
        dados_personagem['Gender_Male'] = [0]

    # Raça
    if caracteristicas[4] == 'Alienigena':
        dados_personagem['Race_group_by_Alien'] = [1]
    else:
        dados_personagem['Race_group_by_Alien'] = [0]
    
    if caracteristicas[4] == 'Seres artificiais':
        dados_personagem['Race_group_by_Artificial Beings'] = [1]
    else:
        dados_personagem['Race_group_by_Artificial Beings'] = [0]

    if caracteristicas[4] == 'Divino ou semidivino':
        dados_personagem['Race_group_by_Divine and Semidivine'] = [1]
    else:
        dados_personagem['Race_group_by_Divine and Semidivine'] = [0]

    if caracteristicas[4] == 'Seres geneticamente alterados':
        dados_personagem['Race_group_by_Genetically Altered Beings'] = [1]
    else:
        dados_personagem['Race_group_by_Genetically Altered Beings'] = [0]

    if caracteristicas[4] == 'Humanos e variações':
        dados_personagem['Race_group_by_Human and Variation'] = [1]
    else:
        dados_personagem['Race_group_by_Human and Variation'] = [0]

    if caracteristicas[4] == 'Mitologico':
        dados_personagem['Race_group_by_Mythical'] = [1]
    else:
        dados_personagem['Race_group_by_Mythical'] = [0]

    if caracteristicas[4] == 'Não identificado':
        dados_personagem['Race_group_by_Not Identified'] = [1]
    else:
        dados_personagem['Race_group_by_Not Identified'] = [0]

    if caracteristicas[4] == 'Subaquatico':
        dados_personagem['Race_group_by_Underwater'] = [1]
    else:
        dados_personagem['Race_group_by_Underwater'] = [0]

    # Estatura
    dados_personagem['Height'] = (caracteristicas[0] - 15.2) / (975 - 15.2)
    dados_personagem['Weight'] = (caracteristicas[1] - 4) / (198416000.0 - 4)

    # Editora
    if caracteristicas[2] == 'DC Comics':
        dados_personagem['Publisher_group_by_DC Comics'] = [1]
    else:
        dados_personagem['Publisher_group_by_DC Comics'] = [0]

    if caracteristicas[2] == 'Marvel Comics':
        dados_personagem['Publisher_group_by_Marvel Comics'] = [1]
    else:
        dados_personagem['Publisher_group_by_Marvel Comics'] = [0]

    if model.predict(dados_personagem)[0] == 'good':
        alinhamento = 'Bom'

    if model.predict(dados_personagem)[0] == 'bad':
        alinhamento = 'Mau'

    if model.predict(dados_personagem)[0] == 'neutral':
        alinhamento = 'Neutro'

    return alinhamento

heroes = pd.read_csv('C:\\Users\\joaop\\Documents\\Minha Carreira Profissional\\Processos Seletivos\\Alelo - Cientista de Dados Pleno\\Resultados\\heroes2_clusterizado.csv').fillna('-')
rf_model = joblib.load('C:\\Users\\joaop\\Documents\\Minha Carreira Profissional\\Processos Seletivos\\Alelo - Cientista de Dados Pleno\\Resultados\\rf_classifier.pkl')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

st.title("Dashboard - Super Herois")

aba1, aba2, aba3 = st.tabs(['Análise Exploratória', 'Análise de Clusters', 'Classificação'])

with aba1:

    df_grafico = heroes[['name', 'Race_group_by', 'Height', 'Weight', 'Gender'] + skills + ['Publisher_group_by', 'Alignment']].copy()
    st.title('Filtros Globais')

    col11, col12 = st.columns(2, border = True)

    with col11:

        filtro_genero = st.popover("Gênero")

        gen_masculino = filtro_genero.checkbox('Male', True)
        gen_feminino = filtro_genero.checkbox('Female', True)
        gen_nidentif = filtro_genero.checkbox('Others / Not Identified', True)

        if not gen_masculino:
            df_grafico = df_grafico[df_grafico.Gender != 'Male']

        if not gen_feminino:
            df_grafico = df_grafico[df_grafico.Gender != 'Female']

        if not gen_nidentif:
            df_grafico = df_grafico[df_grafico.Gender != '-']
            
    with col12:

        filtro_editora = st.popover("Editora")

        editora_marvel = filtro_editora.checkbox('Marvel Comics', True)
        editora_dc = filtro_editora.checkbox('DC Comics', True)
        editora_others = filtro_editora.checkbox('Others', True)

        if not editora_marvel:
            df_grafico = df_grafico[df_grafico.Publisher_group_by != 'Marvel Comics']

        if not editora_dc:
            df_grafico = df_grafico[df_grafico.Publisher_group_by != 'DC Comics']

        if not editora_others:
            df_grafico = df_grafico[df_grafico.Publisher_group_by != 'Others']
        
    st.title('Analises Gerais')

    col11, col12 = st.columns(2, border = True)

    with col11:
        hue_alinhamento = st.toggle("Diferenciar por alinhamento")
        st.pyplot(grafico_raca(df_grafico, hue_alinhamento))

    with col12:
        raca = st.selectbox("Selecione a raça:", df_grafico.Race_group_by.value_counts().index)
        st.pyplot(grafico_skill(df_grafico, raca))

    st.dataframe(df_grafico) 
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

with aba2:

    Cluster = st.selectbox("Selecione o cluster:", heroes.Cluster.drop_duplicates().sort_values().values)
    clusters_data = heroes[heroes.Cluster == Cluster]

    col21, col22 = st.columns(2, border = True)

    with col21:
        st.pyplot(analise_de_cluster(clusters_data, Cluster, 'skill'))
        st.pyplot(analise_de_cluster(clusters_data, Cluster, 'raca'))

    with col22:
        st.pyplot(analise_de_cluster(clusters_data, Cluster, 'genero'))
        st.pyplot(analise_de_cluster(clusters_data, Cluster, 'alinhamento'))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ #

with aba3:

    st.text("Indique a proeficiência do personagem com habilidades")

    col31, col32, col33 = st.columns(3, border = True)

    with col31:

        hab_fisica = st.select_slider("Físicas:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_elemental = st.select_slider("Elemental:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_espaco_tempo = st.select_slider("Espaço-tempo:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_sobrenatural = st.select_slider("Sobrenatural:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_tech = st.select_slider("Tecnológicas:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_sensorial = st.select_slider("Sensoriais e de percepção:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_energia = st.select_slider("Manipulação de energia:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_mental = st.select_slider("Mental e psiquica:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_corporal = st.select_slider("Transformação / transmutação corporal:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_locomocao = st.select_slider("Locomoção:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        hab_combate = st.select_slider("De combate:", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        habilidades = [hab_fisica, hab_energia, hab_elemental, hab_mental, hab_espaco_tempo, hab_corporal, hab_sobrenatural, hab_tech, hab_sensorial, hab_locomocao, hab_combate]

    with col32:

        editora = st.selectbox("Editora:", ['Marvel Comics', 'DC Comics', 'Outras'])
        genero = st.selectbox("Genero:", ['Masculino', 'Feminino', 'Outro / Não Identificado'])
        raca = st.selectbox("Raça", ['Humanos e variações', 'Seres geneticamente alterados', 'Alienigena', 'Seres artificiais', 'Divino ou semidivino', 'Mitologico', 'Subaquatico', 'Não identificado'])
        altura = st.number_input('Altura:')
        peso = st.number_input('Peso:')

        caracteristicas = [altura, peso, editora, genero, raca]

        classificar = st.button('Classificar')

    with col33:
        st.text('O seu personagem foi classificado como:')

        if classificar:
            classificacao_realizada = classificar_personagem(rf_model, habilidades, caracteristicas)

        st.title(classificacao_realizada)

