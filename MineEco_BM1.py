import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Évaluation Économique de Projets Miniers - Métaux de Base",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et en-tête
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Évaluation Économique de Projets Miniers à Ciel Ouvert")
    st.markdown("<h3 style='text-align: center; color: #4F8BF9;'>Cas des Métaux de Bases</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>Développé par Didier Ouedraogo, P.Geo</p>", unsafe_allow_html=True)

# Fonction pour formater les nombres
def format_number(number, decimal_places=0):
    if decimal_places > 0:
        return f"{number:,.{decimal_places}f}".replace(",", " ")
    else:
        return f"{number:,.0f}".replace(",", " ")

# Créer les onglets
tabs = st.tabs([
    "Données d'entrée", 
    "Production Minière", 
    "Flux de Trésorerie", 
    "Indicateurs Financiers", 
    "Répartition de la Rente", 
    "Graphiques d'Analyse"
])

# Onglet 1: Données d'entrée
with tabs[0]:
    st.header("Données d'entrée du projet")
    
    # Données de base
    st.subheader("Données de base")
    col1, col2 = st.columns(2)
    
    with col1:
        nom_minerai = st.text_input("Commodité", placeholder="Ex: Cuivre, Zinc, Plomb...")
        prix_metal = st.number_input("Prix du métal (USD/tonne)", min_value=0.0, help="Prix de marché du métal pur")
    
    with col2:
        duree_projet = st.number_input("Durée du projet (années)", min_value=1, max_value=50, value=10)
        investissement_initial = st.number_input("Investissement initial (USD)", min_value=0)
    
    # Paramètres techniques
    st.subheader("Paramètres techniques")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        teneur_minerai = st.number_input("Teneur du minerai (%)", min_value=0.0, max_value=100.0, step=0.01, 
                                       help="Concentration du métal dans le minerai extrait")
        teneur_concentre = st.number_input("Teneur du concentré (%)", min_value=0.0, max_value=100.0, step=0.01,
                                         help="Concentration du métal dans le produit après traitement")
    
    with col2:
        ratio_sterile = st.number_input("Ratio stérile/minerai", min_value=0.0, step=0.1,
                                      help="Rapport entre la quantité de stérile et la quantité de minerai")
        production_tout_venant = st.number_input("Production annuelle tout venant (tonnes)", min_value=0,
                                              help="Quantité totale de matériau extrait (minerai + stérile)")
    
    with col3:
        # Champs calculés automatiquement
        if teneur_minerai > 0 and teneur_concentre > 0 and ratio_sterile >= 0 and production_tout_venant > 0:
            production_minerai = production_tout_venant / (ratio_sterile + 1)
            facteur_concentration = teneur_concentre / teneur_minerai
            production_concentre = production_minerai / facteur_concentration
            contenu_metal = production_concentre * (teneur_concentre / 100)
            
            st.metric("Production annuelle de minerai (tonnes)", format_number(production_minerai))
            st.metric("Facteur de concentration", f"{facteur_concentration:.2f}")
            st.metric("Production annuelle de concentré (tonnes)", format_number(production_concentre))
            st.metric("Contenu Métal Annuel (tonnes)", format_number(contenu_metal))
        else:
            st.info("Veuillez remplir tous les paramètres techniques pour voir les valeurs calculées")
            production_minerai = 0
            facteur_concentration = 0
            production_concentre = 0
            contenu_metal = 0
    
    # Paramètres économiques
    st.subheader("Paramètres économiques")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cout_extraction = st.number_input("Coût d'extraction du minerai (USD/tonne)", min_value=0.0, step=0.01)
        cout_production = st.number_input("Coût de production du concentré (USD/tonne)", min_value=0.0, step=0.01)
        taux_actualisation = st.number_input("Taux d'actualisation (%)", min_value=0.0, max_value=100.0, step=0.1, value=10.0)
    
    with col2:
        taux_imposition = st.number_input("Taux d'imposition (%)", min_value=0.0, max_value=100.0, step=0.1, value=35.0)
        duree_amortissement = st.number_input("Durée d'amortissement (années)", min_value=1, max_value=30, value=5)
    
    with col3:
        taux_royalty = st.number_input("Taux de royalty (%)", min_value=0.0, max_value=100.0, step=0.1, value=3.0)
        participation_etat = st.number_input("Participation gratuite étatique (%)", min_value=0.0, max_value=100.0, step=0.1, value=10.0)
    
    # Bouton de calcul
    calculate_button = st.button("Calculer", type="primary", use_container_width=True)

# Fonction pour calculer tous les paramètres financiers
def calculate_financial_parameters():
    # Conversion des pourcentages en décimales
    taux_actualisation_decimal = taux_actualisation / 100
    taux_imposition_decimal = taux_imposition / 100
    taux_royalty_decimal = taux_royalty / 100
    participation_etat_decimal = participation_etat / 100
    
    # Calcul de l'amortissement annuel
    amortissement_annuel = investissement_initial / duree_amortissement
    
    # Initialisation des tableaux pour stocker les données annuelles
    annees = list(range(duree_projet + 1))
    recettes = [0] * (duree_projet + 1)
    depenses = [0] * (duree_projet + 1)
    amortissements = [0] * (duree_projet + 1)
    royalties = [0] * (duree_projet + 1)
    benefices_bruts = [0] * (duree_projet + 1)
    impots = [0] * (duree_projet + 1)
    benefices_nets = [0] * (duree_projet + 1)
    flux_tresorerie = [-investissement_initial] + ([0] * duree_projet)
    flux_actualises = [-investissement_initial] + ([0] * duree_projet)
    flux_cumules = [-investissement_initial] + ([0] * duree_projet)
    
    # Calcul des flux de trésorerie pour chaque année
    flux_cumule = -investissement_initial
    somme_total_flux = 0
    somme_flux_actualises = 0
    somme_impots = 0
    somme_royalties = 0
    
    for annee in range(1, duree_projet + 1):
        # Calculs annuels
        recette_annuelle = prix_metal * contenu_metal
        depense_annuelle = (cout_extraction * production_minerai) + (cout_production * production_concentre)
        amortissement_annee = amortissement_annuel if annee <= duree_amortissement else 0
        royalty_annuelle = recette_annuelle * taux_royalty_decimal
        benefice_brut = recette_annuelle - depense_annuelle - amortissement_annee - royalty_annuelle
        impot_annuel = max(0, benefice_brut * taux_imposition_decimal)
        benefice_net = benefice_brut - impot_annuel
        flux_annuel = benefice_net + amortissement_annee
        flux_actualise = flux_annuel / ((1 + taux_actualisation_decimal) ** annee)
        flux_cumule += flux_actualise
        
        # Stockage des résultats
        recettes[annee] = recette_annuelle
        depenses[annee] = depense_annuelle
        amortissements[annee] = amortissement_annee
        royalties[annee] = royalty_annuelle
        benefices_bruts[annee] = benefice_brut
        impots[annee] = impot_annuel
        benefices_nets[annee] = benefice_net
        flux_tresorerie[annee] = flux_annuel
        flux_actualises[annee] = flux_actualise
        flux_cumules[annee] = flux_cumule
        
        # Sommes pour les calculs de répartition
        somme_total_flux += flux_annuel
        somme_flux_actualises += flux_actualise
        somme_impots += impot_annuel
        somme_royalties += royalty_annuelle
    
    # Calcul des indicateurs financiers
    van = somme_flux_actualises - investissement_initial
    
    # Calcul du TRI
    irr = np.irr(flux_tresorerie) if not any(flow >= 0 and flow_next <= 0 for flow, flow_next in zip(flux_tresorerie[:-1], flux_tresorerie[1:])) else None
    
    # Calcul du délai de récupération actualisé
    if flux_actualises[-1] >= 0:
        payback_period = next((i + abs(flux_cumules[i]) / abs(flux_actualises[i+1])) for i in range(len(flux_cumules) - 1) if flux_cumules[i] < 0 and flux_cumules[i+1] >= 0)
    else:
        payback_period = float('inf')
    
    # Calcul du ratio bénéfice/coût
    ratio_benefice_cout = (somme_flux_actualises + investissement_initial) / investissement_initial
    
    # Calcul de la répartition de la rente minière
    participation_etat_valeur = somme_total_flux * participation_etat_decimal
    total_etat = somme_impots + somme_royalties + participation_etat_valeur
    total_prive = somme_total_flux - total_etat
    ratio_repartition = total_etat / total_prive if total_prive != 0 else float('inf')
    
    # Créer le DataFrame pour le tableau des flux
    df_flux = pd.DataFrame({
        'Année': annees,
        'Recettes': recettes,
        'Dépenses': depenses,
        'Amortissement': amortissements,
        'Royalties': royalties,
        'Bénéfice brut': benefices_bruts,
        'Impôts': impots,
        'Bénéfice net': benefices_nets,
        'Flux de trésorerie': flux_tresorerie,
        'Flux actualisé': flux_actualises,
        'Flux cumulé': flux_cumules
    })
    
    return {
        'df_flux': df_flux,
        'van': van,
        'tri': irr * 100 if irr is not None else None,
        'delai_recuperation': payback_period,
        'ratio_benefice_cout': ratio_benefice_cout,
        'total_etat': total_etat,
        'total_prive': total_prive,
        'ratio_repartition': ratio_repartition,
        'somme_impots': somme_impots,
        'somme_royalties': somme_royalties,
        'participation_etat_valeur': participation_etat_valeur,
        'somme_total_flux': somme_total_flux
    }

# Variable globale pour stocker les résultats
results = {}

# Calcul des résultats si le bouton est cliqué
if calculate_button:
    # Vérifier que tous les champs obligatoires sont remplis
    if (nom_minerai and prix_metal > 0 and cout_extraction > 0 and cout_production > 0 and duree_projet > 0 
        and investissement_initial > 0 and teneur_minerai > 0 and teneur_concentre > 0 
        and ratio_sterile >= 0 and production_tout_venant > 0):
        
        results = calculate_financial_parameters()
        st.session_state.results = results
        st.success("Calculs réalisés avec succès!")
        
    else:
        st.error("Veuillez remplir tous les champs correctement.")

# Récupérer les résultats de la session si disponibles
if 'results' in st.session_state:
    results = st.session_state.results

# Onglet 2: Production Minière
with tabs[1]:
    st.header("Données de production")
    
    if results:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Données d'extraction")
            st.metric("Production annuelle tout venant", f"{format_number(production_tout_venant)} tonnes")
            st.metric("Ratio stérile/minerai", f"{ratio_sterile:.2f}")
            st.metric("Production annuelle de minerai", f"{format_number(production_minerai)} tonnes")
            st.metric("Teneur du minerai", f"{teneur_minerai:.2f}%")
        
        with col2:
            st.subheader("Données de traitement")
            st.metric("Teneur du concentré", f"{teneur_concentre:.2f}%")
            st.metric("Facteur de concentration", f"{facteur_concentration:.2f}")
            st.metric("Production annuelle de concentré", f"{format_number(production_concentre)} tonnes")
            st.metric("Contenu Métal Annuel", f"{format_number(contenu_metal)} tonnes")
        
        # Graphique de production
        st.subheader("Visualisation des données de production")
        
        fig = px.bar(
            x=["Tout venant", "Minerai", "Concentré", "Métal"],
            y=[production_tout_venant, production_minerai, production_concentre, contenu_metal],
            labels={"x": "Type de matériau", "y": "Tonnes"},
            title="Comparaison des tonnages",
            color_discrete_sequence=["#1a5276", "#2e86c1", "#3498db", "#85c1e9"],
            text_auto=True
        )
        
        fig.update_layout(
            height=500,
            yaxis=dict(type="log"),
            xaxis=dict(title=""),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez d'abord calculer les résultats dans l'onglet 'Données d'entrée'.")

# Onglet 3: Flux de Trésorerie
with tabs[2]:
    st.header("Tableau des flux de trésorerie (en USD)")
    
    if results:
        # Afficher le tableau des flux
        df_flux = results['df_flux']
        
        # Formater les colonnes numériques
        numeric_cols = df_flux.columns.drop('Année')
        for col in numeric_cols:
            df_flux[col] = df_flux[col].apply(lambda x: format_number(x))
        
        st.dataframe(df_flux, use_container_width=True)
        
        # Option pour télécharger les données
        excel_file = BytesIO()
        df_flux.to_excel(excel_file, index=False, engine="openpyxl")
        excel_file.seek(0)
        
        st.download_button(
            label="Télécharger le tableau Excel",
            data=excel_file,
            file_name=f"flux_tresorerie_{nom_minerai}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Veuillez d'abord calculer les résultats dans l'onglet 'Données d'entrée'.")

# Onglet 4: Indicateurs Financiers
with tabs[3]:
    st.header("Indicateurs financiers clés")
    
    if results:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Valeur Actuelle Nette (VAN)", f"{format_number(results['van'])} USD")
            st.metric("Taux de Rendement Interne (TRI)", f"{results['tri']:.2f}%" if results['tri'] is not None else "Non calculable")
        
        with col2:
            st.metric("Délai de Récupération", f"{results['delai_recuperation']:.2f} années" if results['delai_recuperation'] != float('inf') else "Non récupérable")
            st.metric("Ratio Bénéfice/Coût", f"{results['ratio_benefice_cout']:.2f}")
        
        # Graphique d'interprétation des indicateurs
        st.subheader("Interprétation des indicateurs")
        
        # Viabilité du projet selon la VAN et le TRI
        viabilite = "Très rentable" if results['van'] > 0 and (results['tri'] is None or results['tri'] > taux_actualisation * 2) else \
                   "Rentable" if results['van'] > 0 and (results['tri'] is None or results['tri'] > taux_actualisation) else \
                   "Peu rentable" if results['van'] > 0 else \
                   "Non rentable"
        
        couleur_viabilite = {
            "Très rentable": "#27ae60",
            "Rentable": "#2ecc71",
            "Peu rentable": "#f39c12",
            "Non rentable": "#e74c3c"
        }
        
        st.markdown(f"<div style='background-color: {couleur_viabilite[viabilite]}; padding: 20px; border-radius: 10px;'>"
                   f"<h3 style='color: white; text-align: center;'>Ce projet est: {viabilite}</h3>"
                   "</div>", 
                   unsafe_allow_html=True)
        
        # Comparaison TRI vs Taux d'actualisation
        if results['tri'] is not None:
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=results['tri'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "TRI vs Taux d'actualisation"},
                gauge={
                    'axis': {'range': [0, max(results['tri'] * 1.5, taux_actualisation * 2)]},
                    'bar': {'color': "#2e86c1"},
                    'steps': [
                        {'range': [0, taux_actualisation], 'color': "#e74c3c"},
                        {'range': [taux_actualisation, taux_actualisation * 1.5], 'color': "#f39c12"},
                        {'range': [taux_actualisation * 1.5, max(results['tri'] * 1.5, taux_actualisation * 2)], 'color': "#27ae60"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': taux_actualisation
                    }
                },
                delta={'reference': taux_actualisation, 'relative': True, 'position': "top"}
            ))
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Le TRI n'est pas calculable pour ce projet.")
    else:
        st.info("Veuillez d'abord calculer les résultats dans l'onglet 'Données d'entrée'.")

# Onglet 5: Répartition de la Rente
with tabs[4]:
    st.header("Répartition de la rente minière")
    
    if results:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Part de l'État (Taxes + Royalties + Participation)", 
                f"{format_number(results['total_etat'])} USD", 
                f"{(results['total_etat'] / results['somme_total_flux'] * 100):.2f}%"
            )
            
            st.metric(
                "Part des actionnaires privés", 
                f"{format_number(results['total_prive'])} USD", 
                f"{(results['total_prive'] / results['somme_total_flux'] * 100):.2f}%"
            )
            
            st.metric("Ratio de répartition (État/Privé)", f"{results['ratio_repartition']:.2f}")
        
        with col2:
            # Graphique de répartition État vs Privé
            fig = px.pie(
                values=[results['total_etat'], results['total_prive']],
                names=["Part de l'État", "Part des actionnaires privés"],
                title="Répartition État vs Privé",
                color_discrete_sequence=["#3498db", "#e74c3c"],
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique détaillé des revenus de l'État
        st.subheader("Détail des revenus de l'État")
        
        fig = px.pie(
            values=[results['somme_impots'], results['somme_royalties'], results['participation_etat_valeur']],
            names=["Impôts", "Royalties", "Participation de l'État"],
            title="Composition des revenus de l'État",
            color_discrete_sequence=["#27ae60", "#8e44ad", "#f39c12"]
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Veuillez d'abord calculer les résultats dans l'onglet 'Données d'entrée'.")

# Onglet 6: Graphiques d'Analyse
with tabs[5]:
    st.header("Analyse graphique des flux")
    
    if results:
        # Évolution des flux de trésorerie
        st.subheader("Évolution des flux de trésorerie")
        
        df_flux = results['df_flux']
        
        fig = px.bar(
            df_flux[1:],  # Exclure l'année 0
            x="Année",
            y=["Flux de trésorerie", "Flux actualisé"],
            title="Évolution des flux de trésorerie",
            barmode="group",
            color_discrete_map={
                "Flux de trésorerie": "#3498db",
                "Flux actualisé": "#e74c3c"
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Évolution du flux cumulé
        st.subheader("Évolution du flux cumulé")
        
        fig = px.line(
            df_flux,
            x="Année",
            y="Flux cumulé",
            title="Évolution du flux cumulé actualisé",
            markers=True,
            color_discrete_sequence=["#2ecc71"]
        )
        
        # Ajouter une ligne horizontale à y=0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=duree_projet,
            y1=0,
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Ajouter une annotation pour le délai de récupération si applicable
        if results['delai_recuperation'] != float('inf'):
            fig.add_vline(
                x=results['delai_recuperation'],
                line_dash="dot",
                line_color="#f39c12",
                annotation_text=f"Délai de récupération: {results['delai_recuperation']:.2f} ans",
                annotation_position="top right"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse de sensibilité
        st.subheader("Analyse de sensibilité")
        
        # Choisir les variables à analyser
        col1, col2 = st.columns(2)
        
        with col1:
            sensitivity_variable = st.selectbox(
                "Variable à analyser",
                ["Prix du métal", "Teneur du minerai", "Coût d'extraction", "Taux d'actualisation"]
            )
        
        with col2:
            sensitivity_range = st.slider(
                "Plage de variation (%)",
                min_value=10,
                max_value=50,
                value=30,
                step=5
            )
        
        # Définir la plage de variation
        variation_range = np.linspace(-sensitivity_range/100, sensitivity_range/100, 11)
        
        # Initialiser les listes pour stocker les résultats
        sensitivity_data = []
        
        # Calculer la VAN pour chaque variation
        base_values = {
            "Prix du métal": prix_metal,
            "Teneur du minerai": teneur_minerai,
            "Coût d'extraction": cout_extraction,
            "Taux d'actualisation": taux_actualisation
        }
        
        # Fonction pour recalculer les résultats avec une valeur modifiée
        def recalculate_with_modified_value(variable, variation):
            original_value = base_values[variable]
            modified_value = original_value * (1 + variation)
            
            # Stocker la valeur originale
            if variable == "Prix du métal":
                nonlocal prix_metal
                temp_value = prix_metal
                prix_metal = modified_value
            elif variable == "Teneur du minerai":
                nonlocal teneur_minerai, production_concentre, contenu_metal, facteur_concentration
                temp_value = teneur_minerai
                teneur_minerai = modified_value
                # Recalculer les valeurs dépendantes
                facteur_concentration = teneur_concentre / teneur_minerai
                production_concentre = production_minerai / facteur_concentration
                contenu_metal = production_concentre * (teneur_concentre / 100)
            elif variable == "Coût d'extraction":
                nonlocal cout_extraction
                temp_value = cout_extraction
                cout_extraction = modified_value
            elif variable == "Taux d'actualisation":
                nonlocal taux_actualisation
                temp_value = taux_actualisation
                taux_actualisation = modified_value
            
            # Calculer les nouveaux résultats
            new_results = calculate_financial_parameters()
            
            # Restaurer la valeur originale
            if variable == "Prix du métal":
                prix_metal = temp_value
            elif variable == "Teneur du minerai":
                teneur_minerai = temp_value
                # Recalculer les valeurs dépendantes à nouveau
                facteur_concentration = teneur_concentre / teneur_minerai
                production_concentre = production_minerai / facteur_concentration
                contenu_metal = production_concentre * (teneur_concentre / 100)
            elif variable == "Coût d'extraction":
                cout_extraction = temp_value
            elif variable == "Taux d'actualisation":
                taux_actualisation = temp_value
            
            return new_results
        
        # Calculer pour chaque variation
        for variation in variation_range:
            try:
                new_results = recalculate_with_modified_value(sensitivity_variable, variation)
                
                sensitivity_data.append({
                    "Variation (%)": variation * 100,
                    "VAN": new_results['van'],
                    "TRI (%)": new_results['tri'] if new_results['tri'] is not None else 0
                })
            except:
                st.warning(f"Calcul impossible pour la variation de {variation * 100:.0f}%")
        
        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data)
            
            # Créer le graphique de sensibilité
            fig1 = px.line(
                sensitivity_df,
                x="Variation (%)",
                y="VAN",
                title=f"Sensibilité de la VAN à la variation de {sensitivity_variable}",
                markers=True,
                color_discrete_sequence=["#3498db"]
            )
            
            fig1.add_hline(
                y=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil de rentabilité",
                annotation_position="bottom right"
            )
            
            fig1.add_vline(
                x=0,
                line_dash="dot",
                line_color="#95a5a6",
                annotation_text="Valeur de base",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Graphique pour le TRI si applicable
            if not all(tri == 0 for tri in sensitivity_df["TRI (%)"]):
                fig2 = px.line(
                    sensitivity_df,
                    x="Variation (%)",
                    y="TRI (%)",
                    title=f"Sensibilité du TRI à la variation de {sensitivity_variable}",
                    markers=True,
                    color_discrete_sequence=["#e74c3c"]
                )
                
                fig2.add_hline(
                    y=taux_actualisation,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Taux d'actualisation",
                    annotation_position="bottom right"
                )
                
                fig2.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color="#95a5a6",
                    annotation_text="Valeur de base",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Veuillez d'abord calculer les résultats dans l'onglet 'Données d'entrée'.")

# Pied de page
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<p style='text-align: center;'>Application propulsée par <a href='https://streamlit.io' target='_blank'>Streamlit Cloud</a></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>© 2025 Didier Ouedraogo, P.Geo - Tous droits réservés</p>", unsafe_allow_html=True)