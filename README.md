# 🤖 AI Agent - Fine-tuning

![Banner](https://cdn.pixabay.com/photo/2023/08/15/14/05/banner-8192025_1280.png)

![AI Agent](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)

## 📝 Descrição
Este repositório documenta o desenvolvimento de um **agente de chatbot baseado em IA**, projetado para atuar como **assistente virtual de uma loja de roupas**. O chatbot foi treinado utilizando **Fine-tuning** com a API da OpenAI, permitindo que forneça informações precisas sobre:

📍 **Localização da loja**  
🔄 **Políticas de devolução**  
💳 **Métodos de pagamento aceitos**  
🛍️ **Itens em estoque e variação de modelos/cores**  
📏 **Tamanhos disponíveis**  
🏬 **Sessões da loja**  

Para melhorar a eficiência, foi implementado um modelo preditivo com **scikit-learn** para identificar se a mensagem do usuário está relacionada ao contexto da loja. Isso reduz o consumo de tokens da OpenAI e torna o chatbot mais eficiente.

---

## 🚀 Tecnologias Utilizadas
- 🏗 **[CrewAI](https://github.com/joaomdmoura/crewai)** - Orquestração de agentes de IA
- 🤖 **[OpenAI API](https://openai.com/api/)** - Modelo de IA generativa e Fine-tuning
- 📊 **[scikit-learn](https://scikit-learn.org/)** - Modelo preditivo para otimização do chatbot

---

## 🔧 Funcionalidades
✅ **Assistente virtual**: Capaz de responder perguntas sobre a loja, produtos e serviços.  
📉 **Otimização de Tokens**: Reduz o consumo de tokens da OpenAI ao verificar se a mensagem está dentro do contexto da loja.  
📈 **Aprendizado Contínuo**: O modelo preditivo é atualizado constantemente através de um **sistema de feedback automático**, aumentando sua eficácia ao longo do tempo.  



---

## 📌 Melhorias Futuras
✨ Integração com APIs para facilitar a gestão de estoque e pedidos.  
📊 Implementação de um **dashboard** para monitoramento do chatbot.  
🎭 Expansão do modelo preditivo para detectar **emoções do usuário** e personalizar respostas.  

---

## 🔗 Repositório
Este repositório é mantido por [Pedro Lucca](https://github.com/pedroluccaDEV).  
Confira o projeto no GitHub: [AI-Agent---Fine-tunning](https://github.com/pedroluccaDEV/AI-Agent---Fine-tunning)

