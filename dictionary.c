#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <err.h>
#include <assert.h>

#include <mpi.h>

// shrding : each process create its own dictonary
// for query : ask the value of certain key and send value of key
// use mpi

typedef uint64_t u64; /* portable 64-bit integer */
typedef uint32_t u32; /* portable 32-bit integer */
struct __attribute__((packed)) entry
{
    u32 k;
    u64 v;
}; /* hash table entry */

/***************************** global variables ******************************/

u64 n = 0; /* block size (in bits) */
u64 local_n = 0;
u64 mask; /* this is 2**n - 1 */

u64 dict_size;   /* number of slots in the hash table */
struct entry *A; /* the hash table */

/* (P, C) : two plaintext-ciphertext pairs */
u32 P[2][2] = {{0, 0}, {0xffffffff, 0xffffffff}};
u32 C[2][2];

int my_rank; /* rank of the process */
int p;       /* number of processes */

/************************ tools and utility functions *************************/

double wtime()
{
    struct timeval ts;
    gettimeofday(&ts, NULL);
    return (double)ts.tv_sec + ts.tv_usec / 1E6;
}

// murmur64 hash functions, tailorized for 64-bit ints / Cf. Daniel Lemire
u64 murmur64(u64 x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

/* represent n in 4 bytes */
void human_format(u64 n, char *target)
{
    if (n < 1000)
    {
        sprintf(target, "%" PRId64, n);
        return;
    }
    if (n < 1000000)
    {
        sprintf(target, "%.1fK", n / 1e3);
        return;
    }
    if (n < 1000000000)
    {
        sprintf(target, "%.1fM", n / 1e6);
        return;
    }
    if (n < 1000000000000ll)
    {
        sprintf(target, "%.1fG", n / 1e9);
        return;
    }
    if (n < 1000000000000000ll)
    {
        sprintf(target, "%.1fT", n / 1e12);
        return;
    }
}

/******************************** SPECK block cipher **************************/

#define ROTL32(x, r) (((x) << (r)) | (x >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))

#define ER32(x, y, k) (x = ROTR32(x, 8), x += y, x ^= k, y = ROTL32(y, 3), y ^= x)
#define DR32(x, y, k) (y ^= x, y = ROTR32(y, 3), x ^= k, x -= y, x = ROTL32(x, 8))

void Speck64128KeySchedule(const u32 K[], u32 rk[])
{
    u32 i, D = K[3], C = K[2], B = K[1], A = K[0];
    for (i = 0; i < 27;)
    {
        rk[i] = A;
        ER32(B, A, i++);
        rk[i] = A;
        ER32(C, A, i++);
        rk[i] = A;
        ER32(D, A, i++);
    }
}

// encrypte
void Speck64128Encrypt(const u32 Pt[], u32 Ct[], const u32 rk[])
{
    u32 i;
    Ct[0] = Pt[0];
    Ct[1] = Pt[1];
    for (i = 0; i < 27;)
        ER32(Ct[1], Ct[0], rk[i++]);
}

// decrypte
void Speck64128Decrypt(u32 Pt[], const u32 Ct[], u32 const rk[])
{
    int i;
    Pt[0] = Ct[0];
    Pt[1] = Ct[1];
    for (i = 26; i >= 0;)
        DR32(Pt[1], Pt[0], rk[i--]);
}

/******************************** dictionary ********************************/

/*
 * "classic" hash table for 64-bit key-value pairs, with linear probing.
 * It operates under the assumption that the keys are somewhat random 64-bit integers.
 * The keys are only stored modulo 2**32 - 5 (a prime number), and this can lead
 * to some false positives.
 */
static const u32 EMPTY = 0xffffffff;
static const u64 PRIME = 0xfffffffb;

/* allocate a hash table with `size` slots (12*size bytes) */
void dict_setup(u64 size)
{
    dict_size = size;
    char hdsize[8];
    human_format(dict_size * sizeof(*A), hdsize);
    printf("Dictionary size: %sB\n", hdsize);

    A = malloc(sizeof(*A) * dict_size);
    if (A == NULL)
        err(1, "impossible to allocate the dictionnary");
    for (u64 i = 0; i < dict_size; i++)
        A[i].k = EMPTY;
}

/* Insert the binding key |----> value in the dictionnary */
void dict_insert(u64 key, u64 value)
{
    u64 h = murmur64(key) % dict_size;
    for (;;)
    {
        if (A[h].k == EMPTY)
            break;
        h += 1;
        if (h == dict_size)
            h = 0;
    }
    assert(A[h].k == EMPTY);
    A[h].k = key % PRIME;
    A[h].v = value;
}

/* Query the dictionnary with this `key`.  Write values (potentially)
 *  matching the key in `values` and return their number. The `values`
 *  array must be preallocated of size (at least) `maxval`.
 *  The function returns -1 if there are more than `maxval` results.
 */
int dict_probe(u64 key, int maxval, u64 values[])
{
    u32 k = key % PRIME;
    u64 h = murmur64(key) % dict_size;
    int nval = 0;
    for (;;)
    {
        if (A[h].k == EMPTY)
            return nval;
        if (A[h].k == k)
        {
            if (nval == maxval)
                return -1;
            values[nval] = A[h].v;
            nval += 1;
        }
        h += 1;
        if (h == dict_size)
            h = 0;
    }
}

u64 f(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Ct[2];
    Speck64128Encrypt(P[0], Ct, rk);
    return ((u64)Ct[0] ^ ((u64)Ct[1] << 32)) & mask;
}

int main(int argc, char **argv)
{

    MPI_Status status;
    /* Initialisation */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int n = 10;
    mask = (1 << n) - 1;

    int size = 1ull << n;

    int local_size = size / p;
    int surplus = size % p;

    // a changer
    if (my_rank == p - 1)
    {
        local_size += surplus;
    }

    dict_setup(1.125 * local_size);

    int debut = my_rank * (size / p);

    u64 *cles = malloc(local_size * sizeof(u64));
    u64 *valeures = malloc(local_size * sizeof(u64));

    int i = 0;
    for (int x = debut; x < debut + local_size; x++)
    {
        u64 z = f(x);
        cles[i] = z;
        valeures[i] = x;
        i++;
    }

    if (my_rank == 0)
    {
        for (int i = 0; i < local_size; i++)
        {
            printf("cle=%ld val=%ld\n", cles[i], valeures[i]);
        }
    }

    // Trier les clés et les valeurs par propriétaire
    int *owner_counts = calloc(p, sizeof(int));
    int *owner_counts2 = calloc(p, sizeof(int));
    u64 **owner_keys = malloc(p * sizeof(u64 *));
    u64 **owner_values = malloc(p * sizeof(u64 *));

    // on compte combien de cle on doit envoyer a un process
    for (int i = 0; i < local_size; i++)
    {
        int owner = cles[i] % p;
        if (owner != my_rank)
        {
            owner_counts[owner]++;
            owner_counts2[owner]++;
        }
    }

    if (my_rank == 0)
    {
        for (int i = 0; i < p; i++)
        {
            printf("proc %d a %d cle\n", i, owner_counts[i]);
        }
    }

    // allocation de la bonne taille par processus
    for (i = 0; i < p; i++)
    {
        owner_keys[i] = malloc(owner_counts[i] * sizeof(u64));
        owner_values[i] = malloc(owner_counts[i] * sizeof(u64));
    }

    // on remplit avec les cle/valeures
    for (i = 0; i < local_size; i++)
    {
        int owner = cles[i] % p;

        if (owner != my_rank)
        {
            owner_keys[owner][owner_counts[owner] - 1] = cles[i];
            owner_values[owner][owner_counts[owner] - 1] = valeures[i];

            owner_counts[owner]--;
        }
    }

    if (my_rank == 0)
    {
        printf("\n\n\n");
        for (int i = 0; i < owner_counts2[2]; i++)
        {
            printf("proc 2 a cle =%ld value %ld et mod =%ld\n", owner_keys[2][i], owner_values[2][i], owner_keys[2][i] % p);
        }
        printf("\n\n\n");
    }

     // Envoyer les clés et les valeurs au bon propriétaire
    for (i = 0; i < p; i++)
    {
        if (i == my_rank)
            continue;
        if (owner_counts2[i] > 0)
        {
            for(int a=0; a<owner_counts2[i];a++){
                printf(" envoi %ld %ld\n",owner_keys[i][a], owner_values[i][a]);
            }
            MPI_Send(owner_keys[i], owner_counts2[i], MPI_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Send(owner_values[i], owner_counts2[i], MPI_LONG, i, 0, MPI_COMM_WORLD);
        }
    }

    // Recevoir les clés et les valeurs des autres processus
    for (i = 0; i < p; i++)
    {
        if (i == my_rank)
            continue;
        int count;
        MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_LONG, &count);
        u64 *recv_keys = malloc(count * sizeof(u64));
        u64 *recv_values = malloc(count * sizeof(u64));
        MPI_Recv(recv_keys, count, MPI_LONG, i, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(recv_values, count, MPI_LONG, i, 0, MPI_COMM_WORLD, &status);

        // Insérer les clés et les valeurs reçues dans le dictionnaire local
        for (int j = 0; j < count; j++)
        {  
            printf("insert %ld %ld\n",recv_keys[j], recv_values[j]);
            dict_insert(recv_keys[j], recv_values[j]);
        }

        free(recv_keys);
        free(recv_values);
    }

    if (my_rank == 3)
    {
        for (int i = 0; i < local_size; i++)
        {
            printf("cle=%d val=%ld et %d\n", A[i].k, A[i].v, A[i].k % p);
        }
    }

    // Libérer la mémoire
    for (i = 0; i < p; i++)
    {
        free(owner_keys[i]);
        free(owner_values[i]);
    }
    free(owner_keys);
    free(owner_values);
    free(owner_counts);
    free(cles);
    free(valeures);

    MPI_Finalize();

    return 0;
}
