#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <mpi.h>
#include <sstream>
#include <string>

using namespace std;

void update(std::vector<int>& block, std::vector<int>& leftBlock, std::vector<int>& rightBlock, int fieldSize, int size) {
    std::vector<int> swapBlock(block.size());
    int x, y, xx, yy, count, lineCount;

    lineCount = block.size() / fieldSize;

    for (int i = 0; i < block.size(); i++) {
        count = 0;
        x = i / fieldSize;
        y = i % fieldSize;

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dy == 0 and dx == 0)
                    continue;

                yy = (fieldSize + dy + y) % fieldSize;

                if (size > 1 && dx + x < 0)
                    count += leftBlock[yy];
                else if (size > 1 && dx + x >= lineCount)
                    count += rightBlock[yy];
                else {
                    xx = (fieldSize + dx + x) % fieldSize;

                    count += block[xx * fieldSize + yy];
                }
            }
        }

        if (count == 3)
            swapBlock[i] = 1;
        else if (count == 2)
            swapBlock[i] = block[i];
        else
            swapBlock[i] = 0;
    }

    block.swap(swapBlock);
}

class Worker {
public:

    Worker(int rank, int workerCount, bool isMaster) {
        _fieldSize = 0;
        _stepCount = 0;
        _rank = rank;
        _workerCount = workerCount;
        _isMaster = isMaster;
        _tag = 1;
    }

    void Start() {
        clock_t start;
        if (_isMaster) {
            start = clock();
            SetupWorkers("setup.txt");
        }
        else
            Setup();

        for (int s = 0; s < _stepCount; s++) {
            Update();
        }

        if (_isMaster) {
            clock_t end = clock();
            printf("time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
        }

        if (_isMaster)
            CollectData();
        else
            MPI_Send(_block.data(), _block.size(), MPI_INT, 0, _tag, MPI_COMM_WORLD);

    }

private:
    vector<int> _block;
    int _rank;
    int _fieldSize;
    int _stepCount;
    int _workerCount;
    bool _isMaster;

    int _tag;

    void SetupWorkers(string file) {
        ifstream infile(file);
        string line;
        getline(infile, line);
        _stepCount = stoi(line);
        getline(infile, line);
        _fieldSize = stoi(line);

        vector<int> field(_fieldSize * _fieldSize);

        for (int y = 0; y < _fieldSize; y++) {
            if (!getline(infile, line))
                break;
            for (int x = 0; x < _fieldSize; x++) {
                if (x * 2 < line.size())
                    field[x * _fieldSize + y] = line[x * 2] - '0';
                else
                    break;
            }
        }

        int lineCount = _fieldSize / _workerCount + (int)(0 < _fieldSize % _workerCount);
        _block = vector<int>(field.begin(), field.begin() + lineCount * _fieldSize);

        int shift = lineCount * _fieldSize;

        for (int i = 1; i < _workerCount; i++) {
            int data[2] = { _fieldSize, _stepCount };

            MPI_Send(data, 2, MPI_INT, i, _tag, MPI_COMM_WORLD);

            lineCount = _fieldSize / _workerCount + (int)(i < _fieldSize % _workerCount);
            vector<int> block = vector<int>(field.begin() + shift, field.begin() + shift + lineCount * _fieldSize);

            MPI_Send(block.data(), block.size(), MPI_INT, i, _tag, MPI_COMM_WORLD);
            
            shift += lineCount * _fieldSize;
        }
    }

    void Setup() {
        int data[2];
        MPI_Status status;

        MPI_Recv(data, 2, MPI_INT, 0, _tag, MPI_COMM_WORLD, &status);

        _fieldSize = data[0];
        _stepCount = data[1];

        int lineCount = _fieldSize / _workerCount + (int)(_rank < _fieldSize % _workerCount);

        _block = vector<int>(_fieldSize * lineCount);

        MPI_Recv(_block.data(), _block.size(), MPI_INT, 0, _tag, MPI_COMM_WORLD, &status);
    }

    void Update() {
        vector<int> leftBlock(_fieldSize);
        vector<int> rightBlock(_fieldSize);

        ExchangeData(leftBlock, rightBlock);

        update(_block, leftBlock, rightBlock, _fieldSize, _workerCount);
    }

    void ExchangeData(vector<int>& leftBlock, vector<int>& rightBlock) {
        MPI_Status status;
        MPI_Sendrecv(_block.data() + _block.size() - _fieldSize, _fieldSize, MPI_INT, (_rank + 1) % _workerCount, _tag,
            leftBlock.data(), leftBlock.size(), MPI_INT, (_rank - 1 + _workerCount) % _workerCount, _tag,
            MPI_COMM_WORLD, &status);
        MPI_Sendrecv(_block.data(), _fieldSize, MPI_INT, (_rank - 1 + _workerCount) % _workerCount, _tag,
            rightBlock.data(), rightBlock.size(), MPI_INT, (_rank + 1) % _workerCount, _tag,
            MPI_COMM_WORLD, &status);
    }

    void CollectData() {
        MPI_Status status;
        vector<int> sumBlock;

        sumBlock.insert(sumBlock.end(), _block.begin(), _block.end());

        for (int i = 1; i < _workerCount; i++) {
            int lineByBlock = _fieldSize / _workerCount + (int)(i < _fieldSize % _workerCount);
            vector<int> gettedBlock(_fieldSize * lineByBlock);
            MPI_Recv(gettedBlock.data(), gettedBlock.size(), MPI_INT, i, _tag, MPI_COMM_WORLD, &status);

            sumBlock.insert(sumBlock.end(), gettedBlock.begin(), gettedBlock.end());
        }

        fstream file;
        file.open("output.txt", ios_base::out);

        for (int y = 0; y < _fieldSize; y++) {
            for (int x = 0; x < _fieldSize; x++)
                file << sumBlock[x * _fieldSize + y] << ' ';
            file << endl;
        }

        file.close();
    }
};

int main(int argc, char** argv) {
    int rank, size, len, tag = 1;
    char host[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(host, &len);

    Worker worker = Worker(rank, size, rank == 0);

    worker.Start();

    MPI_Finalize();
    return 0;
}
