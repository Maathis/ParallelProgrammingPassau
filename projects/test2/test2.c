#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "debug.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_size%2 != 0)
    {
        printf("Please enter an even number of processes\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 0;
    }

    int world_rank, partner_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank%2 == 0)
    {
        partner_rank = world_rank+1;
    } else {
        partner_rank = world_rank-1;
    }

    char msg[13];
    if (world_rank%2 == 0) {
        strcpy(msg, "Hello World!");
        MPI_Send(&msg, 13, MPI_CHAR, partner_rank, 2, MPI_COMM_WORLD);
        printTimingMsg();
        printf("%d send the message to %d\n", world_rank, partner_rank);
    } else {
        MPI_Recv(&msg, 13, MPI_CHAR, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printTimingMsg();
        printf("%d received a message from %d : %s\n", world_rank, partner_rank, msg);
    }

    MPI_Finalize();
    return 0;
}