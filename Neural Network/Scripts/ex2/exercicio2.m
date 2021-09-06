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
learning_rate = 0.2;
iteracoes_maximas = 3000;

%escolha da função de ativação 'TANH' ou 'SIGMOID'
funcao_ativacao = 'TANH';

if contains(funcao_ativacao, 'TANH')
    
    %normalização para a entrada [-1, 1]
    normaliza_entrada = @(x, min, max) ((x-min)/(max-min) * 2 - 1);
    
    %a saída já se encontra normalizada
    normaliza_saida = @(y) y;
    
    %atribuição da função de avaliação
    funcao_ativacao = @(x) tanh(x);
    
    %calculo da derivada 
    derivada_funcao_ativacao = @( x ) sech(x)^2 ;
    
    %função para calcular a classificação da saída
    funcao_classificacao = @(y) (y>0.0)*2-1;
    
    %função para gerar os pesos
    gera_pesos = @(m,n) (rand(m, n) * 2 - 1)/4;
    
    %função para gerar os bias iniciais
    gera_bias = @(m, n) (rand(m, n) * 2 - 1)/4;
    
    %limites dos inputs
    input_minimo = -1;
    input_maximo = 1;   
elseif contains(funcao_ativacao, 'SIGMOID')
    
    %normalização para a entrada [0, 1]
    normaliza_entrada = @(x, min, max) ((x-min)/(max-min));
    
    %normalização para a saída
    normaliza_saida = @(y_des) (y_des + 1)/2;
    
    %atribuição da função de avaliação
    funcao_ativacao = @(x) 1/(1+exp(-x));
    
    %calculo da derivada 
    derivada_funcao_ativacao = @(x) exp(-x)/(exp(-x) + 1)^2;
    
    %função para calcular a classificação da saída
    funcao_classificacao = @(y) (y>0.5) * 2 - 1;
    
    %função para gerar os pesos
    gera_pesos = @(m, n) rand(m, n)/4;
    
    %função para gerar os bias iniciais
    gera_bias = @(m, n) rand(m, n)/4;
    
    %limites dos inputs
    input_minimo = 0;
    input_maximo = 1;   
end

%leitura dos ficheiros
dados_input = csvread('testInput11A.txt');
dados_output = csvread('testOutput11A.txt');

%encontrar a linha onde ocorre o valor (0, 0, 0)
linhas=ismember(dados_input, [0 0 0], 'rows');

%encontrar o tamanho dos dados para treino(tamanho = (linha do elemento 0 0 0) -1
tamanho_treino = find(linhas)-1;

%encontrar o tamanho dos dados para teste
tamanho_teste = length(dados_input) - find(linhas);

%separação dos dados de entrada em dados de treino e dados de teste
dados_treino = dados_input(1:tamanho_treino, :);
dados_teste =  dados_input(tamanho_treino+2: length(dados_input), 1:2);

%criação das matrizes para armazenamento dos dados
matriz_output_rede = zeros(tamanho_treino, 1);
matriz_input_treino = zeros(tamanho_treino, 2);
matriz_output_treino = zeros(tamanho_treino, 1);

matriz_input_teste = zeros(tamanho_teste, 2);
matriz_output_classificado = zeros(tamanho_teste, 1);

%normalização dos valores e preenchimento das matrizes
matriz_input_treino(:,1) = normaliza_entrada(dados_treino(:,1), min(dados_input(:,1)), max(dados_input(:,1)));
matriz_input_treino(:,2) = normaliza_entrada(dados_treino(:,2), min(dados_input(:,2)), max(dados_input(:,2)));

matriz_input_teste(:,1) = normaliza_entrada(dados_teste(:,1), min(dados_input(:,1)), max(dados_input(:,1)));
matriz_input_teste(:,2) = normaliza_entrada(dados_teste(:,2), min(dados_input(:,1)), max(dados_input(:,1)));

matriz_output_treino = normaliza_saida(dados_treino(:,3));

%pesos e bias aleatórios
bias_1 = gera_bias(numero_neuronios_1_camada, 1);
peso_01 = gera_pesos(numero_neuronios_1_camada, 2);

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

numero_erros = tamanho_treino;
iteracoes=0;
count_it = 0;
while iteracoes < iteracoes_maximas && numero_erros>0
    
    %forward propagation
    
    %primeira camada escondida
    for i = 1:tamanho_treino
        
        z1 = peso_01*[matriz_input_treino(i,1); matriz_input_treino(i,2)]+bias_1;
        
        y1 = arrayfun(funcao_ativacao, z1);
        
        z2 = peso_12*y1 + bias_2;
        
        y2 = arrayfun(funcao_ativacao, z2);
        
        z3 = peso_23*y2 + bias_3;
        
        matriz_output_rede(i, 1) = arrayfun(funcao_ativacao, z3);
    
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
        
        peso_23(1,1) = peso_23(1,1) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(1,1);
        peso_23(1,2) = peso_23(1,2) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(2,1);
        peso_23(1,3) = peso_23(1,3) + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1)) * derivada_funcao_ativacao(z3)*y2(3,1);
        bias_3 = bias_3 + learning_rate*(matriz_output_treino(i,1)-matriz_output_rede(i,1))*derivada_funcao_ativacao(z3);
        
    end
    
    
    
    %classificação da saida
    
    numero_erros = 0;
    
    classificacao_rede = funcao_classificacao(matriz_output_rede);
    
    for i=1:tamanho_treino
        if classificacao_rede(i,1) ~= dados_treino(i,3) 
           numero_erros=numero_erros + 1; 
        end
    end
    
    iteracoes = iteracoes + 1;
end

fprintf('Erros no treino = %d\n', numero_erros);

%aplicação da rede neuronal aos dados de teste

numero_erros = 0;

for i=1:tamanho_teste
   
    %forward propagation
    z1 = peso_01*[matriz_input_teste(i,1); matriz_input_teste(i,2)] + bias_1;
    y1 = arrayfun(funcao_ativacao, z1);
    
    z2 = peso_12*y1+bias_2;
    y2 = arrayfun(funcao_ativacao, z2);
    
    z3 = peso_23*y2+bias_3;
    
    matriz_output_rede(i,1)=arrayfun(funcao_ativacao,z3);
    
    classificacao_rede = funcao_classificacao(matriz_output_rede);
    
     if classificacao_rede(i,1) ~= dados_output(i,1) 
          numero_erros=numero_erros + 1; 
     end
    
end

fprintf('Erros no teste = %d || taxa de acerto = %d\n', numero_erros, (tamanho_teste-numero_erros)*100/tamanho_teste);

%plot dos gráficos resultantes do treino e teste

vermelhos = []; %+1
azuis = []; %-1


%adiciona os valores de treino
for i=1:tamanho_treino
    
   if dados_input(i,3) == -1
       vermelhos = [vermelhos; dados_input(i, 1:2)];
   else 
       azuis = [azuis; dados_input(i, 1:2)];
   end
   
end

%adiciona os valores de teste

for i=1:tamanho_teste
    
   if dados_output(i,1) == -1
       vermelhos = [vermelhos; dados_teste(i, 1:2)];
   else 
       azuis = [azuis; dados_teste(i, 1:2)];
   end
   
end

%normalizar novamente

vermelhos(:,1) = normaliza_entrada(vermelhos(:,1), min(dados_input(:,1)), max(dados_input(:,1)));
vermelhos(:,2) = normaliza_entrada(vermelhos(:,2), min(dados_input(:,2)), max(dados_input(:,2)));

azuis(:,1) = normaliza_entrada(azuis(:,1), min(dados_input(:,1)), max(dados_input(:,1)));
azuis(:,2) = normaliza_entrada(azuis(:,2), min(dados_input(:,2)), max(dados_input(:,2)));

%plot

figure(1);
hold on
janela = input_minimo:0.0001:input_maximo;
xlim([input_minimo input_maximo])
ylim([input_minimo input_maximo])


scatter(vermelhos(:,1), vermelhos(:,2), 25, 'r', 'filled')
scatter(azuis(:,1), azuis(:,2), 25, 'b', 'filled')
scatter(matriz_input_teste(:,1), matriz_input_teste(:,2), 25, 'k', 'filled')


xlabel('X')
ylabel('Y')
grid on
legend('classe -1', 'classe +1', 'pontos de teste')

