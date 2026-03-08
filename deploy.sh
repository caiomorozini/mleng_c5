#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_message() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_warning "docker-compose não encontrado, tentando usar 'docker compose'"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

print_message "Usando: $DOCKER_COMPOSE"
print_message "Parando containers existentes..."
$DOCKER_COMPOSE down

print_message "Construindo imagem Docker..."
$DOCKER_COMPOSE build

if [ $? -ne 0 ]; then
    print_error "Falha no build da imagem!"
    exit 1
fi

print_message "Imagem construída com sucesso!"


print_message "Iniciando containers..."
$DOCKER_COMPOSE up -d

if [ $? -ne 0 ]; then
    print_error "Falha ao iniciar containers!"
    exit 1
fi

print_message "Aguardando API inicializar..."
sleep 5

print_message "Verificando saúde da API..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_message "API está saudável!"
        break
    fi

    if [ $i -eq 10 ]; then
        print_error "API não respondeu após 10 tentativas"
        print_message "Logs do container:"
        docker logs datathon
        exit 1
    fi

    sleep 2
done

echo ""
echo "======================================================================"
echo -e "${GREEN}  API DEPLOYADA COM SUCESSO!${NC}"
echo "======================================================================"
echo ""
echo "Endpoints disponíveis:"
echo "   - API Root: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - Documentação: http://localhost:8000/docs"
echo "   - Model Info: http://localhost:8000/model-info"
echo "   - Predict: http://localhost:8000/predict"
echo ""
echo "Comandos úteis:"
echo "   - Ver logs: docker logs -f datathon"
echo "   - Parar: docker-compose down"
echo "   - Reiniciar: docker-compose restart"
echo ""
echo "======================================================================"
