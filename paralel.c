#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <windows.h>

// === Estruturas Globais ===
typedef struct
{
    float **xtreino;
    float **xteste;
    float **matriz_distancias;
    int linhas_treino;
    int linhas_teste;
    int colunas;
} ArgumentosDistancia;

typedef struct
{
    float **matriz_distancias;
    float *ytreino;
    float *classificacoes;
    int linhas_treino;
    int linhas_teste;
    int k;
} ArgumentosClassificacao;

// === Funções Matemáticas Auxiliares ===

double get_time_in_seconds() {
    LARGE_INTEGER frequency, start;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    return (double)start.QuadPart / frequency.QuadPart;
}

float quadrado(float num)
{
    return num * num;
}

float raiz_quadrada(float num)
{
    if (num < 0)
    {
        printf("Erro: número negativo.\n");
        return -1;
    }

    float estimativa = num / 2.0f;
    float epsilon = 0.00001f;

    while ((estimativa * estimativa - num) > epsilon || (num - estimativa * estimativa) > epsilon)
    {
        estimativa = (estimativa + num / estimativa) / 2.0f;
    }

    return estimativa;
}

// === Funções de Inicialização de Dados ===

float **alocar_matriz(int linhas, int colunas)
{
    float **matriz = (float **)malloc(linhas * sizeof(float *));
    for (int i = 0; i < linhas; i++)
    {
        matriz[i] = (float *)malloc(colunas * sizeof(float));
    }
    return matriz;
}

void liberar_matriz(float **matriz, int linhas)
{
    for (int i = 0; i < linhas; i++)
    {
        free(matriz[i]);
    }
    free(matriz);
}

float *alocar_vetor(int tamanho)
{
    return (float *)malloc(tamanho * sizeof(float));
}

void liberar_vetor(float *vetor)
{
    free(vetor);
}

void ler_matriz_teste_arquivo(const char *nome_arquivo, float ***matriz, int *linhas, int colunas)
{
    FILE *arquivo = fopen(nome_arquivo, "r");
    if (!arquivo)
    {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }

    // Carregar todos os valores do arquivo para um vetor
    float *vetor = NULL;
    int total_valores = 0;
    float valor;

    while (fscanf(arquivo, "%f", &valor) == 1)
    {
        vetor = realloc(vetor, (total_valores + 1) * sizeof(float));
        if (!vetor)
        {
            perror("Erro ao alocar memória para vetor");
            fclose(arquivo);
            exit(EXIT_FAILURE);
        }
        vetor[total_valores++] = valor;
    }
    fclose(arquivo);

    // Determinar o número de linhas na matriz
    *linhas = total_valores - colunas + 1;
    if (*linhas <= 0)
    {
        printf("Número de colunas inválido para o tamanho do vetor.\n");
        free(vetor);
        exit(EXIT_FAILURE);
    }

    // Alocar memória para a matriz
    *matriz = malloc(*linhas * sizeof(float *));
    if (!*matriz)
    {
        perror("Erro ao alocar memória para matriz");
        free(vetor);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *linhas; i++)
    {
        (*matriz)[i] = malloc(colunas * sizeof(float));
        if (!(*matriz)[i])
        {
            perror("Erro ao alocar memória para linha da matriz");
            for (int k = 0; k < i; k++)
            {
                free((*matriz)[k]);
            }
            free(*matriz);
            free(vetor);
            exit(EXIT_FAILURE);
        }
    }

    // Preencher a matriz com valores deslocados do vetor
    for (int i = 0; i < *linhas; i++)
    {
        for (int j = 0; j < colunas; j++)
        {
            (*matriz)[i][j] = vetor[i + j];
        }
    }

    free(vetor);
}

void inicializar_vetor(float *vetor, float **matriz, int linhas, int colunas, int h)
{
    for (int i = h; i < linhas; i++) // Começa da segunda linha (índice 1)
    {
        vetor[i - h] = matriz[i][colunas - 1]; // Preenche o vetor com o valor da última coluna
    }
}

// === Funções Principais ===

void calcular_distancias_sequencial(ArgumentosDistancia *dados)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < dados->linhas_teste; i++)
    {
        for (int j = 0; j < dados->linhas_treino; j++)
        {
            float soma = 0.0f;
            for (int k = 0; k < dados->colunas; k++)
            {
                soma += quadrado(dados->xteste[i][k] - dados->xtreino[j][k]);
            }
            dados->matriz_distancias[i][j] = raiz_quadrada(soma);
        }
    }
}

int *encontrar_menores_posicoes(float *vetor_distancias, int tamanho, int k)
{
    int *posicoes = (int *)malloc(sizeof(int) * k);
    float *copia_distancias = (float *)malloc(sizeof(float) * tamanho);

    for (int i = 0; i < tamanho; i++)
    {
        copia_distancias[i] = vetor_distancias[i];
    }

    for (int i = 0; i < k; i++)
    {
        int menor_indice = i;
        for (int j = i + 1; j < tamanho; j++)
        {
            if (copia_distancias[j] < copia_distancias[menor_indice])
            {
                menor_indice = j;
            }
        }
        float temp = copia_distancias[i];
        copia_distancias[i] = copia_distancias[menor_indice];
        copia_distancias[menor_indice] = temp;

        posicoes[i] = menor_indice;
    }

    free(copia_distancias);
    return posicoes;
}

void classificar_knn_sequencial(ArgumentosClassificacao *dados)
{
    #pragma omp parallel for
    for (int i = 0; i < dados->linhas_teste; i++)
    {
        int *menores = encontrar_menores_posicoes(dados->matriz_distancias[i], dados->linhas_treino, dados->k);

        float soma_classes = 0.0;
        for (int b = 0; b < dados->k; b++)
        {
            soma_classes += dados->ytreino[menores[b]];
        }
        dados->classificacoes[i] = soma_classes / dados->k;

        free(menores);
    }
}

// === Função Principal ===

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        fprintf(stderr, "Uso: %s <arquivo_xtreino> <arquivo_xteste> <colunas>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *arquivo_xtreino = argv[1];
    const char *arquivo_xteste = argv[2];
    int colunas = atoi(argv[3]);
    int k = atoi(argv[4]);
    int h = atoi(argv[5]);

    if (colunas <= 0)
    {
        fprintf(stderr, "Erro: o número de colunas deve ser maior que zero.\n");
        return EXIT_FAILURE;
    }

    int linhas_treino;
    int linhas_teste;

    float **xtreino;
    float **xteste;

    ler_matriz_teste_arquivo(arquivo_xtreino, &xtreino, &linhas_treino, colunas);
    ler_matriz_teste_arquivo(arquivo_xteste, &xteste, &linhas_teste, colunas);

    float *ytreino = alocar_vetor(linhas_treino);
    float **matriz_distancias = alocar_matriz(linhas_teste, linhas_treino);
    float *classificacoes = (float *)malloc(linhas_teste * sizeof(float));

    inicializar_vetor(ytreino, xtreino, linhas_treino, colunas, h);

    double inicio = get_time_in_seconds();

    // Calculando as distâncias sequencialmente
    ArgumentosDistancia argumentos_distancias = {xtreino, xteste, matriz_distancias, linhas_treino, linhas_teste, colunas};
    calcular_distancias_sequencial(&argumentos_distancias);

    // Classificando sequencialmente
    ArgumentosClassificacao argumentos_classificacao = {matriz_distancias, ytreino, classificacoes, linhas_treino, linhas_teste, k};
    classificar_knn_sequencial(&argumentos_classificacao);

    double fim = get_time_in_seconds();

    FILE *arquivo_ytreino = fopen("ytreino.txt", "w");
    if (arquivo_ytreino == NULL)
    {
        perror("Erro ao abrir o arquivo para salvar ytreino");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < linhas_treino; i++)
    {
        fprintf(arquivo_ytreino, "%f\n", ytreino[i]);
    }
    fclose(arquivo_ytreino);


    FILE *arquivo = fopen("resultado_classificacoes.txt", "w");
    if (arquivo == NULL)
    {
        perror("Erro ao abrir o arquivo");
        return 1; // Saia do programa se não for possível abrir o arquivo
    }

    for (int i = 0; i < linhas_teste; i++)
    {
        fprintf(arquivo, "Classe estimada para a linha %d de xteste: %f\n", i + 1, classificacoes[i]);
    }

    fprintf(arquivo, "Tempo de Execução: %f segundos", (double)(fim - inicio));

    fclose(arquivo);

    // Liberando memória
    liberar_matriz(xtreino, linhas_treino);
    liberar_matriz(xteste, linhas_teste);
    liberar_matriz(matriz_distancias, linhas_teste);
    liberar_vetor(ytreino);
    free(classificacoes);

    return 0;
}
