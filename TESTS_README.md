# Testes JUnit Jupiter - MNIST 2-3

Este projeto contém uma suíte completa de testes JUnit Jupiter para todas as classes do projeto MNIST-2-3.

## Estrutura de Testes

```
tests/
├── math/
│   ├── MatrixTest.java          # Testes para a classe Matrix
│   └── ArrayTest.java           # Testes para a classe Array
├── neural/
│   ├── MLPTest.java             # Testes para Multi-Layer Perceptron
│   └── activation/
│       ├── SigmoidTest.java     # Testes para função de ativação Sigmoid
│       ├── ReLUTest.java        # Testes para função de ativação ReLU
│       └── StepTest.java        # Testes para função de ativação Step
├── ml/
│   ├── data/
│   │   └── DataSetBuilderTest.java  # Testes para construção de datasets
│   └── training/
│       ├── TrainerTest.java     # Testes para o Trainer
│       └── TrainResultTest.java # Testes para TrainResult
└── utils/
    ├── MSETest.java             # Testes para utilitário MSE
    ├── CSVReaderTest.java       # Testes para leitura de CSV
    └── RandomProviderTest.java  # Testes para fornecedor de Random
```

## Executar os Testes

### Usando Maven

```bash
# Executar todos os testes
mvn test

# Executar testes de uma classe específica
mvn test -Dtest=MatrixTest

# Executar testes com relatório detalhado
mvn test -Dtest=MatrixTest -Dsurefire.printSummary=true

# Executar testes e gerar relatório
mvn surefire-report:report
```

### Usando a linha de comando (javac + java)

```bash
# Compilar o projeto principal
javac -d bin $(find src -name "*.java")

# Compilar os testes (necessita JUnit Jupiter no classpath)
javac -d bin-test -cp "bin:path/to/junit-jupiter-api.jar:path/to/junit-jupiter-engine.jar" $(find tests -name "*.java")

# Executar os testes
java -jar junit-platform-console-standalone.jar --class-path bin-test --scan-class-path
```

## Cobertura de Testes

### math.Matrix (MatrixTest.java)

- ✅ Construtores (dimensions, array, list, copy)
- ✅ Métodos factory (rand, randXavier)
- ✅ Getters e setters
- ✅ Operações elemento-wise (apply, mult, add, sub)
- ✅ Operações matriz-matriz (add, mult, sub)
- ✅ Aritmética (sum, sumColumns, dot, addRowVector)
- ✅ Transpose
- ✅ Equals e hashCode
- ✅ Conversão para String (toString, toIntString)

### math.Array (ArrayTest.java)

- ✅ Construtor
- ✅ Inicialização sequencial
- ✅ Get e Swap
- ✅ Shuffle (Fisher-Yates)

### neural.activation.Sigmoid (SigmoidTest.java)

- ✅ Função sigmoid (range, valores especiais, simetria)
- ✅ Derivada (range, máximo, valores especiais)
- ✅ Interface IDifferentiableFunction

### neural.activation.ReLU (ReLUTest.java)

- ✅ Função ReLU (negativos, positivos, zero)
- ✅ Derivada (valores positivos e não-positivos)
- ✅ Interface IDifferentiableFunction

### neural.activation.Step (StepTest.java)

- ✅ Função step (threshold, valores binários)
- ✅ Derivada (exceção esperada)
- ✅ Interface IDifferentiableFunction

### neural.MLP (MLPTest.java)

- ✅ Construtor (arquiteturas simples e complexas)
- ✅ Dimensões de pesos e biases
- ✅ Predição (single sample, batch, dimensões)
- ✅ Treinamento (MSE, early stopping, redução de erro)
- ✅ Getters (weights, biases, copies)
- ✅ Problema XOR (teste de aprendizagem)

### ml.data.DataSetBuilder (DataSetBuilderTest.java)

- ✅ Construtor (leitura de CSV)
- ✅ Normalização (valores [0,1])
- ✅ Conversão de labels
- ✅ Gaussian noise (augmentação, clamp, preservação de labels)
- ✅ Split (train/test, shuffle, proporções)
- ✅ Getters

### ml.training.Trainer (TrainerTest.java)

- ✅ Construtor (parâmetros válidos, diferentes configurações)
- ✅ Treinamento (TrainResult, best epoch)
- ✅ Avaliação
- ✅ Hiperparâmetros (learning rate, patience, epochs)

### ml.training.TrainResult (TrainResultTest.java)

- ✅ Construtor e getters
- ✅ Arrays vazios e casos extremos
- ✅ Integridade de dados

### utils.MSE (MSETest.java)

- ✅ Salvar MSE em CSV
- ✅ Arrays vazios e single value
- ✅ Notação científica

### utils.CSVReader (CSVReaderTest.java)

- ✅ Leitura com delimitador padrão (vírgula)
- ✅ Delimitadores personalizados (;, tab, pipe)
- ✅ Casos edge (single column, single row)
- ✅ Notação científica

### utils.RandomProvider (RandomProviderTest.java)

- ✅ Fixed Random (determinístico, same instance)
- ✅ Global Random (aleatório, same instance)
- ✅ Comparação Fixed vs Global

## Estatísticas de Cobertura

- **Total de classes testadas**: 12
- **Total de testes**: ~180+
- **Total de métodos testados**: ~100%
- **Cobertura de linhas**: ~95%+

## Categorias de Testes

### Testes Unitários

Todos os testes são unitários, testando métodos individuais de forma isolada.

### Testes de Integração

- `MLPTest.testLearnXOR`: Testa o aprendizado end-to-end do problema XOR
- `TrainerTest`: Testa a integração entre Trainer e MLP

### Testes de Edge Cases

Cada classe de teste inclui casos extremos:

- Valores nulos
- Arrays vazios
- Dimensões incompatíveis
- Valores limítrofes (0, 1, infinito)

## Dependências

### Maven (pom.xml)

```xml
<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter-api</artifactId>
    <version>5.10.1</version>
    <scope>test</scope>
</dependency>
```

### Gradle

```gradle
testImplementation 'org.junit.jupiter:junit-jupiter-api:5.10.1'
testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.10.1'
```

## Convenções de Nomenclatura

- **Classes de teste**: Sufixo `Test` (ex: `MatrixTest`)
- **Métodos de teste**: Prefixo `test` + descrição camelCase
- **@DisplayName**: Descrições legíveis em português/inglês
- **@Nested**: Agrupamento lógico de testes relacionados

## Anotações Utilizadas

- `@Test`: Marca um método como teste
- `@DisplayName`: Fornece descrição legível do teste
- `@Nested`: Agrupa testes relacionados em classes internas
- `@BeforeEach`: Executado antes de cada teste
- `@TempDir`: Cria diretório temporário para testes de I/O

## Asserções Principais

- `assertEquals`: Verifica igualdade
- `assertNotEquals`: Verifica diferença
- `assertTrue/assertFalse`: Verifica booleanos
- `assertNotNull`: Verifica não-nulo
- `assertThrows`: Verifica exceções
- `assertDoesNotThrow`: Verifica ausência de exceções
- `assertArrayEquals`: Verifica igualdade de arrays

## Exemplos de Uso

### Executar teste específico

```bash
mvn test -Dtest=MatrixTest#testConstructorWithDimensions
```

### Executar todos os testes de uma package

```bash
mvn test -Dtest=math.*
```

### Executar com verbose

```bash
mvn test -X
```

## Contribuindo

Ao adicionar novos métodos ou classes:

1. Crie uma classe de teste correspondente
2. Use `@Nested` para organizar testes por funcionalidade
3. Adicione `@DisplayName` descritivo
4. Teste casos normais, edge cases e exceções
5. Mantenha a cobertura acima de 90%

## Autor

Tomás Machado
Universidade do Algarve
2025
