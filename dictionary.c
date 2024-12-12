#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <getopt.h>
#include <err.h>
#include <assert.h>
#include <unistd.h>

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

int local_size;
int size;
int debut;
int local_size_dico;

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
    {
        A[i].k = EMPTY;
        A[i].v = 0;
    }
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
    if (my_rank == 0)
    {
        // printf("insert en %ld\n", h);
    }

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

void affiche_dico()
{

    fflush(stdout);
    printf("rank=%d\n", my_rank);
    for (int i = 0; i < local_size_dico; i++)
    {
        printf("%u, %ld\n", A[i].k, A[i].v);
    }
    fflush(stdout);
}

u64 g(u64 k)
{
    assert((k & mask) == k);
    u32 K[4] = {k & 0xffffffff, k >> 32, 0, 0};
    u32 rk[27];
    Speck64128KeySchedule(K, rk);
    u32 Pt[2];
    Speck64128Decrypt(Pt, C[0], rk);
    return ((u64)Pt[0] ^ ((u64)Pt[1] << 32)) & mask;
}

void remplit_dico()
{
    MPI_Status status;
    MPI_Request request;
    debut = my_rank * (size / p);

    u64 *cles = malloc(local_size * sizeof(u64));
    u64 *valeures = malloc(local_size * sizeof(u64));

    int i = 0;
    for (int x = debut; x < debut + local_size; x++)
    {
        u64 z = f(x);
        cles[i] = z;
        valeures[i] = x;
        i++;

        if (z % p == my_rank)
        {
            // //printf("insert\n");
            dict_insert(z, x);
        }
    }
    // //printf("\n\n\n\n\n");
    // affiche_dico();
    // //printf("\n\n\n\n\n");

    if (my_rank == 1)
    {
        for (int i = 0; i < local_size; i++)
        {
            printf("cle=%ld val=%ld, mod=%ld\n", cles[i], valeures[i],cles[i]%p);
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
            // //printf("proc %d a %d cle\n", i, owner_counts[i]);
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
        // //printf("\n\n\n");
        for (int i = 0; i < owner_counts2[2]; i++)
        {
            // //printf("proc 2 a cle =%ld value %ld et mod =%ld\n", owner_keys[2][i], owner_values[2][i], owner_keys[2][i] % p);
        }
        // //printf("\n\n\n");
    }

    u64 rien_a_envoyer = EMPTY;

    // Envoyer les clés et les valeurs au bon propriétaire
    for (i = 0; i < p; i++)
    {
        if (i == my_rank)
            continue;
        if (owner_counts2[i] > 0)
        {
            for (int a = 0; a < owner_counts2[i]; a++)
            {
                if (my_rank == 0)
                {
                    // //printf(" envoi %ld %ld\n", owner_keys[i][a], owner_values[i][a]);
                }
            }
            //printf("%d envoi a %d\n", my_rank, i);
            MPI_Send(owner_keys[i], owner_counts2[i], MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Send(owner_values[i], owner_counts2[i], MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
        else
        {
            if (my_rank == 0)
            {
                // //printf("envoi rien a %d\n", i);
            }
            //printf("%d envoi rien a %d\n", my_rank, i);

            MPI_Send(&rien_a_envoyer, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
            MPI_Send(&rien_a_envoyer, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
        }
    }

    //printf("\n\n");

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

        if (recv_keys[0] != rien_a_envoyer)
        {
            //printf("%d recoi de %d\n", my_rank, i);
            // Insérer les clés et les valeurs reçues dans le dictionnaire local
            for (int j = 0; j < count; j++)
            {
                // //printf("insert %ld %ld\n",recv_keys[j], recv_values[j]);
                dict_insert(recv_keys[j], recv_values[j]);
            }
            //printf("%d a finit d'inserer\n", my_rank);
        }
        else
        {
            //printf("%d recoi rien de %d\n", my_rank, i);
        }

        free(recv_keys);
        free(recv_values);
    }
    //printf("\n\n");

    //printf("les envois sont finits pour %d !!!!!\n", my_rank);

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
}

bool is_good_pair(u64 k1, u64 k2)
{
    u32 Ka[4] = {k1 & 0xffffffff, k1 >> 32, 0, 0};
    u32 Kb[4] = {k2 & 0xffffffff, k2 >> 32, 0, 0};
    u32 rka[27];
    u32 rkb[27];
    Speck64128KeySchedule(Ka, rka);
    Speck64128KeySchedule(Kb, rkb);
    u32 mid[2];
    u32 Ct[2];
    Speck64128Encrypt(P[1], mid, rka);
    Speck64128Encrypt(mid, Ct, rkb);
    return (Ct[0] == C[1][0]) && (Ct[1] == C[1][1]);
}

int golden_claw_search(int maxres, u64 k1[], u64 k2[])
{
    double start = wtime();
    remplit_dico();
    double mid = wtime();

    printf("Fill: %.1fs\n", mid - start);

    if (my_rank == 0)
    {
        printf(" dico rang %d \n", my_rank);
        for (int i = 0; i < local_size; i++)
        {
            printf("rank=%d,cle=%d val=%ld et %d\n", my_rank, A[i].k, A[i].v, A[i].k % p);
        }
    }

    int nres = 0;
    u64 candidates = 0;
    u64 x[256];

    u64 rcvBuff;

    u64 requestBuff;

    MPI_Status status;
    MPI_Request request;

    int flagDemande;
    int flagRcv;

    for (u64 z = debut; z < debut + local_size; z++)
    {
        u64 y = g(z);
        int owner = y % p;

        if (owner == my_rank)
        {
            int nx = dict_probe(y, 256, x);
            assert(nx >= 0);
            candidates += nx;
            for (int i = 0; i < nx; i++)
            {
                if (is_good_pair(x[i], z))
                {
                    if (nres == maxres)
                    {
                        return -1;
                    }
                    k1[nres] = x[i];
                    k2[nres] = z;
                    printf("SOLUTION FOUND!\n");
                    nres += 1;
                }
            }
        }
        else
        {
            MPI_Send(&y, 1, MPI_UNSIGNED_LONG, owner, 1, MPI_COMM_WORLD);
            MPI_Irecv(x, 256, MPI_UNSIGNED_LONG, owner, 1, MPI_COMM_WORLD, &request);

            // Boucle pour vérifier la réception
            while (!flagRcv)
            {
                MPI_Test(&request, &flagRcv, &status);
                if (flagRcv)
                {
                    int nx = status.MPI_TAG;
                    candidates += nx;
                    for (int i = 0; i < nx; i++)
                    {
                        if (is_good_pair(x[i], z))
                        {
                            if (nres == maxres)
                            {
                                return -1;
                            }
                            k1[nres] = x[i];
                            k2[nres] = z;
                            printf("SOLUTION FOUND!\n");
                            nres += 1;
                        }
                    }
                }
                else
                {
                    // Faire d'autres travaux ici si nécessaire
                    MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flagDemande, &status);
                    if (flagDemande)
                    {
                        u64 requestBuff;
                        MPI_Recv(&requestBuff, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
                        int nx = dict_probe(requestBuff, 256, x);
                        MPI_Send(x, nx, MPI_UNSIGNED_LONG, status.MPI_SOURCE, nx, MPI_COMM_WORLD);
                    }
                }
            }
        }

        // Vérifier si une requête a été reçue d'un autre processus
        MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flagDemande, &status);
        if (flagDemande)
        {
            u64 requestBuff;
            MPI_Recv(&requestBuff, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
            int nx = dict_probe(requestBuff, 256, x);
            MPI_Send(x, nx, MPI_UNSIGNED_LONG, status.MPI_SOURCE, nx, MPI_COMM_WORLD);
        }
    }

    printf("Probe: %.1fs. %" PRId64 " candidate pairs tested\n", wtime() - mid, candidates);

    return nres;
}

/************************** command-line options ****************************/

void usage(char **argv)
{
    printf("%s [OPTIONS]\n\n", argv[0]);
    printf("Options:\n");
    printf("--n N                       block size [default 24]\n");
    printf("--C0 N                      1st ciphertext (in hex)\n");
    printf("--C1 N                      2nd ciphertext (in hex)\n");
    printf("\n");
    printf("All arguments are required\n");
    exit(0);
}

void process_command_line_options(int argc, char **argv)
{
    struct option longopts[4] = {
        {"n", required_argument, NULL, 'n'},
        {"C0", required_argument, NULL, '0'},
        {"C1", required_argument, NULL, '1'},
        {NULL, 0, NULL, 0}};
    char ch;
    int set = 0;
    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
    {
        switch (ch)
        {
        case 'n':
            n = atoi(optarg);
            mask = (1ull << n) - 1;
            break;
        case '0':
            set |= 1;
            u64 c0 = strtoull(optarg, NULL, 16);
            C[0][0] = c0 & 0xffffffff;
            C[0][1] = c0 >> 32;
            break;
        case '1':
            set |= 2;
            u64 c1 = strtoull(optarg, NULL, 16);
            C[1][0] = c1 & 0xffffffff;
            C[1][1] = c1 >> 32;
            break;
        default:
            errx(1, "Unknown option\n");
        }
    }
    if (n == 0 || set != 3)
    {
        usage(argv);
        exit(1);
    }
}

int main(int argc, char **argv)
{

    /* Initialisation */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    process_command_line_options(argc, argv);
    // printf("Running with n=%d, C0=(%08x, %08x) and C1=(%08x, %08x)\n",
    //        (int)n, C[0][0], C[0][1], C[1][0], C[1][1]);

    // int n = 10;
    mask = (1 << n) - 1;

    size = 1ull << n;

    local_size = size / p;
    int surplus = size % p;

    // a changer
    if (my_rank == p - 1)
    {
        local_size += surplus;
    }

    local_size_dico = local_size + surplus;

    // printf("my_rank =%d, p=%d , size=%d, local_size=%d, surplus =  %d\n", my_rank, p, size, local_size, surplus);

    /// !!!!!!!!!!!!!!!!!! la taille des dico peut etre bloquant
    dict_setup(1.125 * local_size_dico+2);

    remplit_dico();
    sleep(my_rank / 2);
    affiche_dico();

    // u64 k1[16], k2[16];
    //
    //// remplit_dico();
    // int nkey = golden_claw_search(16, k1, k2);
    //
    // for (int i = 0; i < nkey; i++)
    //{
    //    assert(f(k1[i]) == g(k2[i]));
    //    assert(is_good_pair(k1[i], k2[i]));
    //    printf("Solution found: (%" PRIx64 ", %" PRIx64 ") [checked OK]\n", k1[i], k2[i]);
    //}

    MPI_Finalize();

    return 0;
}