import json
import argparse
from crewai import Agent, Crew, Task
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Defina sua API Key do OpenAI aqui
API_KEY = "YOU-API-KEY"
FINE_TUNED_MODEL = "FINE-TUNNING-OUTPUT-MODEL"

# Carregar dados do treinamento de limites
with open('related.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['question'] for item in data]
labels = [item['label'] for item in data]

# Dividir dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Criar o pipeline de treinamento com ajustes
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),  # Captura unigramas e bigramas
    MultinomialNB(alpha=1.0)  # Suavização padrão
)

model.fit(X_train, y_train)


# Avaliar o modelo
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Função que identifica se uma pergunta é relacionada ou não
def is_related_question(question):
    return model.predict([question])[0] == 'related'

# Função para criar agentes a partir de definições JSON
def create_agents(agents_json, llm):
    agents = []
    for agent_def in agents_json:
        agent = Agent(
            role=agent_def["role"],
            goal=agent_def["goal"],
            verbose=agent_def.get("verbose", False),
            memory=agent_def.get("memory", False),
            backstory=agent_def["backstory"],
            llm=llm
        )
        agents.append(agent)
    return agents

# Função para criar tarefas a partir de solicitações dinâmicas do usuário
def create_task_from_prompt(agent, user_prompt):
    # Criar uma nova tarefa com base no prompt do usuário
    task_description = f"Responder à solicitação do usuário: {user_prompt}"
    task = Task(
        description=task_description,
        expected_output="Resposta personalizada para a solicitação do usuário.",
        agent=agent
    )
    return task

# Função para processar o prompt do usuário e responder com base no modelo
def get_response_from_model(prompt, llm):
    response = llm.invoke(prompt)
    if isinstance(response, dict):
        return response.get('content', '')
    elif hasattr(response, 'text'):
        return response.text
    else:
        return str(response)

# Função para gerar uma saudação personalizada
def generate_welcome_message(agent):
    return f"Eu sou o {agent.role}, seu assistente de vendas aqui na loja ABC. Estou aqui para ajudar com informações sobre nossos produtos, promoções e políticas. Como posso ajudar você hoje?"

# Função para gravar feedback em related.json
def save_feedback(feedback):
    # Carregar o conteúdo existente do arquivo ou criar uma lista vazia se o arquivo não existir ou estiver vazio
    try:
        with open('related.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            if not isinstance(existing_data, list):
                existing_data = []  # Corrigir se o arquivo contiver dados não esperados
    except FileNotFoundError:
        existing_data = []  # Se o arquivo não existir, criar uma lista vazia
    except json.JSONDecodeError:
        existing_data = []  # Se o arquivo estiver vazio ou malformado, criar uma lista vazia
    
    # Adicionar o novo feedback à lista
    existing_data.append({"question": feedback, "label": "related"})
    
    # Salvar o conteúdo atualizado de volta no arquivo
    with open('related.json', 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)  # Usar ensure_ascii=False para preservar caracteres Unicode

# Função principal para processar argumentos, carregar JSON e criar agentes e tarefas
def main():
    parser = argparse.ArgumentParser(description="Process CrewAI agent and task definitions.")
    parser.add_argument('--json', type=str, help="JSON string with agent and task definitions")
    parser.add_argument('--file', type=str, help="Path to a JSON file with agent and task definitions")

    args = parser.parse_args()

    if args.json:
        definitions = json.loads(args.json)
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as json_file:
            definitions = json.load(json_file)
    else:
        raise ValueError("Either --json or --file must be provided.")

    llm = ChatOpenAI(
        model=FINE_TUNED_MODEL,  # Use o model_id do modelo fine-tuned
        api_key=API_KEY
    )

    agents = create_agents(definitions["agents"], llm)
    tasks = []  # Iniciar com tarefas vazias

    # Inicializar o Crew
    crew = Crew(agents=agents, tasks=tasks)
    print(f"Crew initialized with {len(agents)} agents and {len(tasks)} tasks.")

    # Selecionar o primeiro agente para interagir com o usuário
    agent = agents[0] if agents else None

    if agent:
        welcome_message = generate_welcome_message(agent)
        print(welcome_message)
    else:
        print("Nenhum agente disponível para interação.")

    # Memória de conversa para manter o contexto
    conversation_history = []

    while True:
        user_prompt = input("Digite sua pergunta (ou 'sair' para encerrar): ")
        if user_prompt.lower() == 'sair':
            print("Até logo! Se precisar de ajuda, estarei aqui.")
            break

        if user_prompt.lower() == 'feedback':
            feedback = input("Digite seu feedback: ")
            save_feedback(feedback)
            print("Obrigado pelo seu feedback!")
            continue

        conversation_history.append({"role": "user", "content": user_prompt})

        if is_related_question(user_prompt):
            # Criar uma nova tarefa com base na solicitação do usuário
            task = create_task_from_prompt(agent, user_prompt)
            tasks.append(task)

            # Atualizar o Crew com a nova tarefa
            crew.tasks = tasks
            try:
                # Reiniciar o Crew para considerar a nova tarefa
                crew.kickoff()
                # Limpar a lista de tarefas após a execução para evitar múltiplas tarefas
                tasks.clear()
            except ValueError as e:
                print(f"Error during kickoff: {e}")

            # Adicionar contexto da conversa no prompt
            context_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            response = get_response_from_model(context_prompt, llm)
            conversation_history.append({"role": "agent", "content": response})
        else:
            response = "Desculpe, eu só posso responder perguntas relacionadas a nossa loja e aos nossos produtos."
            conversation_history.append({"role": "agent", "content": response})

if __name__ == "__main__":
    main()
