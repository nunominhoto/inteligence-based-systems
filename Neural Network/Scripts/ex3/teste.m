%exercicio 2

close all
clear all
clc

%defenição do número de neurónios da rede
numero_neuronios_1_camada = 5;
numero_neuronios_2_camada = 3;
numero_neuronios_saida = 1;
numero_neuronios_entrada = 2;

%outros parâmetros
% para 'TANH' melhor learning_rate = 0.1
% para 'SIGMOID' melhor rate = 0.2
learning_rate = 0.1;
iteracoes_maximas = 6000;

%escolha da função de ativação 'TANH' ou 'SIGMOID'
funcao_ativacao = 'TANH';

if contains(funcao_ativacao, 'TANH')
    
    %normalização para a entrada [-1, 1]
    normaliza_entrada = @(x, min, max) ((x-min)/(max-min) * 2 - 1);
    
    %[-1, 1]
    normaliza_saida = @(y, min, max) ((y-min)/(max-min) * 2 - 1);
    
    %
    desnormaliza = @(f, min, max) ((f+1)/2*(max-min)+min);
    
    %atribuição da função de avaliação
    funcao_ativacao = @(x) tanh(x);
    
    %calculo da derivada 
    derivada_funcao_ativacao = @( x ) sech(x)^2 ;
    
    %função para calcular a classificação da saída
    funcao_classificacao = @(y) (y>0.0)*2-1;
    
    %função para gerar os pesos
    gera_pesos = @(m,n) (rand(m, n) * 2 - 1)/4.5;
    
    %função para gerar os bias iniciais
    gera_bias = @(m, n) (rand(m, n) * 2 - 1)/4.5;
    
    %limites dos inputs
    input_minimo = -1;
    input_maximo = 1;   
end
    
%leitura dos ficheiros
dados_input = csvread('DataSet3.txt');
%dados_output = csvread('testOutput11B.txt');

%encontrar a linha onde ocorre o valor (0, 0, 0)
linhas=ismember(dados_input, [0 0 0], 'rows');

%encontrar o tamanho dos dados para treino(tamanho = (linha do elemento 0 0 0) -1
tamanho_treino = find(linhas)-1;

%encontrar o tamanho dos dados para teste
tamanho_teste = length(dados_input) - find(linhas);

%separação dos dados de entrada em dados de treino e dados de teste
dados_treino = dados_input(1:tamanho_treino, :);
dados_teste =  dados_input(tamanho_treino+2: length(dados_input), :);

%criação das matrizes para armazenamento dos dados
matriz_output_rede = zeros(tamanho_treino, 1);
matriz_input_treino = zeros(tamanho_treino, 2);
matriz_output_treino = zeros(tamanho_treino, 1);

matriz_input_teste = zeros(tamanho_teste, 2);
matriz_output_classificado = zeros(tamanho_teste, 1);

%normalização dos valores e preenchimento das matrizes

matriz_input_treino(:,1) = normaliza_entrada(dados_treino(:,1), 50, 1000);
matriz_input_treino(:,2) = normaliza_entrada(dados_treino(:,2), -40, 52);

matriz_input_teste(:,1) = normaliza_entrada(dados_teste(:,1), 50, 1000);
matriz_input_teste(:,2) = normaliza_entrada(dados_teste(:,2), -40, 52);

matriz_output_treino = normaliza_saida(dados_treino(:,3), min(dados_treino(:,3)), max(dados_treino(:,3)));
matriz_output_classificado = dados_teste(:,3);

%pesos e bias aleatórios
bias_1 = gera_bias(numero_neuronios_1_camada, 1);
peso_01 = gera_pesos(numero_neuronios_1_camada, 2);%pesos transpostos

bias_2 = gera_bias(numero_neuronios_2_camada, 1);
peso_12 = gera_pesos(numero_neuronios_2_camada, numero_neuronios_1_camada);

bias_3 = gera_bias(numero_neuronios_saida, 1);
peso_23 = gera_pesos(numero_neuronios_saida, numero_neuronios_2_camada);

y1 = zeros(numero_neuronios_1_camada,1);
y2 = zeros(numero_neuronios_2_camada,1);
y3 = zeros(numero_neuronios_saida,1);

z1 = zeros(numero_neuronios_1_camada,1);
z2 = zeros(numero_neuronios_2_camada,1);
z3 = zeros(numero_neuronios_saida,1);

%treino

iteracoes=0;

error = zeros(tamanho_treino,1);
MSE = 100;

while iteracoes < iteracoes_maximas
    
    %forward propagation
    
    %primeira camada escondida
    for i = 1:tamanho_treino
        
        %forward pass
        z1 = peso_01*[matriz_input_treino(i,1); matriz_input_treino(i,2)]+bias_1;
        
        y1 = arrayfun(funcao_ativacao, z1);%saidas dos neuronios da primeira camada escondidada
        
        z2 = peso_12*y1 + bias_2;
        
        y2 = arrayfun(funcao_ativacao, z2);%saida da segunda camada escondida
        
        z3 = peso_23*y2 + bias_3;
        
        matriz_output_rede(i, 1) = arrayfun(funcao_ativacao, z3);%output
        
        error(i,1) = matriz_output_rede(i, 1) - matriz_output_treino(i, 1);
    
             %backwards propagation
        %percorrer os neuronios da primeira camada
        for neuronio=1 : numero_neuronios_1_camada

            %atualiza os pesos de 0 para 1                                  
            peso_01(neuronio, 1) = peso_01(neuronio, 1) + learning_rate*(matriz_output_treino(i, 1) - matriz_output_rede(i, 1))*derivada_funcao_ativacao(z3(1,1))*derivada_funcao_ativacao(z1(neuronio,1))*matriz_input_treino(i,1)*(peso_23(1,1)*derivada_funcao_ativacao(z2(1,1))*peso_12(1,1)+peso_23(1,2)*derivada_funcao_ativacao(z2(2,1))*peso_12(2,1)+peso_23(1,3)*derivada_funcao_ativacao(z2(3,1))*peso_12(3,1));
            peso_01(neuronio, 2) = peso_01(neuronio, 2) + learning_rate*(matriz_output_treino(i, 1) - matriz_output_rede(i, 1))*derivada_funcao_ativacao(z3(1,1))*derivada_funcao_ativacao(z1(neuronio,1))*matriz_input_treino(i,2)*(peso_23(1,1)*derivada_funcao_ativacao(z2(1,1))*peso_12(1,1)+peso_23(1,2)*derivada_funcao_ativacao(z2(2,1))*peso_12(2,1)+peso_23(1,3)*derivada_funcao_ativacao(z2(3,1))*peso_12(3,1));
            %atualiza bias_1
            bias_1(neuronio, 1) = bias_1(neuronio, 1) + learning_rate * (matriz_output_treino(i,1) - matriz_output_rede(i,1))*derivada_funcao_ativacao(z3(1, 1))*derivada_funcao_ativacao(z1(neuronio,1))*1*(peso_23(1,1)*derivada_funcao_ativacao(z2(1,1))*peso_12(1,1)+peso_23(1,2)*derivada_funcao_ativacao(z2(2,1))*peso_12(2,1)+peso_23(1,3)*derivada_funcao_ativacao(z2(3,1))*peso_12(3,1));
        end
        
             %segunda camada escondida
        for neuronio=1:numero_neuronios_2_camada

            for n=1:numero_neuronios_1_camada

                peso_12(neuronio, n) = peso_12(neuronio, n) + learning_rate*(matriz_output_treino(i,1) - matriz_output_rede(i,1))*derivada_funcao_ativacao(z3)*peso_23(1, neuronio)*derivada_funcao_ativacao(z2(neuronio,1))*y1(n,1);
            end

            %atualiza bias_2:
            bias_2(neuronio,1) = bias_2(neuronio,1) + learning_rate*(matriz_output_treino(i,1) - matriz_output_rede(i, 1))*derivada_funcao_ativacao(z3)*peso_23(1, neuronio)*derivada_funcao_ativacao(z2(neuronio,1));

        end
    
        %atualiza pesos camada de saída
        %erro = 1/2*(target-output)^2
        %erro = 1/2*(target^2-2*target*output+output^2)
        %derro/output = -(target-output)
                                                    %derro/dout
                                                    %*dout/dy *dy/dw
        peso_23(1,1) = peso_23(1,1) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(1,1);
        peso_23(1,2) = peso_23(1,2) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(2,1);
        peso_23(1,3) = peso_23(1,3) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(3,1);
        bias_3 = bias_3 + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1))*derivada_funcao_ativacao(z3);
        
    end
    
    
    %calculo do MSE
    error = (error.^2)/tamanho_treino;
    MSE = sum(error);
    
    iteracoes = iteracoes + 1;
end

clear matriz_output_rede;
matriz_output_rede = zeros(tamanho_teste,1);
fprintf("O MSE do treino foi: %d\n", MSE);
MSE = 0;
error_teste = zeros(tamanho_teste,1);
%aplicação da rede neuronal aos dados de teste

%desnormalizacao
for i=1:tamanho_teste
   
    %forward propagation
    z1 = peso_01*[matriz_input_teste(i,1); matriz_input_teste(i,2)] + bias_1;
    y1 = arrayfun(funcao_ativacao, z1);
    
    z2 = peso_12*y1+bias_2;
    y2 = arrayfun(funcao_ativacao, z2);
    
    z3 = peso_23*y2+bias_3;
    
    matriz_output_rede(i,1)=arrayfun(funcao_ativacao,z3);
    
    matriz_output_rede(i,1) = desnormaliza(matriz_output_rede(i,1), min(dados_treino(:,3)), max(dados_treino(:,3)));%saida
    error_teste(i,1) = matriz_output_rede(i, 1) - matriz_output_classificado(i, 1);%erro
end

abs(error_teste./matriz_output_classificado)*100
matriz_output_rede
error_teste = (error_teste.^2)/tamanho_teste;
MSE = sum(error_teste);

fprintf("O MSE do teste foi: %d\n", MSE);





