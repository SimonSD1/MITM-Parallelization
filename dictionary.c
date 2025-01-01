// Simon Sepiol-Duchemin, Joshua Setia

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
#include <time.h>
#include <string.h>
#include <sys/resource.h>
#include <mpi.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

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

int surplus;

/* (P, C) : two plaintext-ciphertext pairs */
u32 P[2][2] = {{0, 0}, {0xffffffff, 0xffffffff}};
u32 C[2][2];

int my_rank; /* rank of the process */
int p;       /* number of processes */

int local_size;
int size;
int debut;
int nb_chunk;
int fixed_size;

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

/* allocate a hash table with size slots (12*size bytes) */
void dict_setup(u64 size)
{
    dict_size = size;
    char hdsize[8];
    human_format(dict_size * sizeof(*A), hdsize);
    if (my_rank == 0)
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

/* Query the dictionnary with this key.  Write values (potentially)
 *  matching the key in values and return their number. The values
 *  array must be preallocated of size (at least) maxval.
 *  The function returns -1 if there are more than maxval results.
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
    for (int i = 0; i < dict_size; i++)
    {
        if (A[i].k != EMPTY)
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
            {
                u64 c0 = strtoull(optarg, NULL, 16);
                C[0][0] = c0 & 0xffffffff;
                C[0][1] = c0 >> 32;
            }
            break;
        case '1':
            set |= 2;
            {
                u64 c1 = strtoull(optarg, NULL, 16);
                C[1][0] = c1 & 0xffffffff;
                C[1][1] = c1 >> 32;
            }
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

void print_memory_usage()
{
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // *** printf("Memory usage: %ld KB\n", usage.ru_maxrss); // ***
}

void remplit_dico()
{
    dict_setup(1.125 * (size / p ) + 100);

    debut = my_rank * (size / p);

    // Trier les clés et les valeurs par propriétaire
    u64 *owner_keys = malloc(p * fixed_size * sizeof(u64));
    u64 *owner_values = malloc(p * fixed_size * sizeof(u64));

    u64 *recv_keys = calloc(p * fixed_size, sizeof(u64));
    u64 *recv_values = calloc(p * fixed_size, sizeof(u64));

    for (int num = 0; num < nb_chunk; num++)
    {
        int *owner_offsets = calloc(p, sizeof(int));

        int debut_chunk = debut + (local_size / nb_chunk) * num;
        int fin_chunk = debut + (local_size / nb_chunk) * (num + 1) + (num == nb_chunk - 1 ? local_size % nb_chunk : 0);

        for (int x = debut_chunk; x < fin_chunk; x++)
        {
            u64 z = f(x);
            int owner = z % p;
            owner_keys  [owner * fixed_size + owner_offsets[owner] + 1] = z;
            owner_values[owner * fixed_size + owner_offsets[owner] + 1] = x;
            owner_offsets[owner]++;
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < p; i++)
        {
            owner_keys  [i * fixed_size] = owner_offsets[i];
            owner_values[i * fixed_size] = owner_offsets[i];
        }

        MPI_Alltoall(owner_keys, fixed_size, MPI_UNSIGNED_LONG,
                     recv_keys, fixed_size, MPI_UNSIGNED_LONG,
                     MPI_COMM_WORLD);
        MPI_Alltoall(owner_values, fixed_size, MPI_UNSIGNED_LONG,
                     recv_values, fixed_size, MPI_UNSIGNED_LONG,
                     MPI_COMM_WORLD);

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < p; i++)
        {
            if (i == my_rank)
            {
                for (int j = 0; j < owner_offsets[my_rank]; j++)
                {
                    dict_insert(owner_keys [my_rank * fixed_size + j],
                                owner_values[my_rank * fixed_size + j]);
                }
            }
            else
            {
                int count = recv_keys[i * fixed_size];
                for (int j = 1; j <= count; j++)
                {
                    dict_insert(recv_keys  [i * fixed_size + j],
                                recv_values[i * fixed_size + j]);
                }
            }
        }

        free(owner_offsets);
    }

    // *** print_memory_usage(); // ***
    free(recv_keys);
    free(recv_values);
    free(owner_keys);
    free(owner_values);
}

int golden_claw_search(int maxres, u64 k1[], u64 k2[])
{
    //double start = wtime();
    remplit_dico();
    //double mid = wtime();

    // *** if (my_rank == 0) printf("Fill: %.001fs\n", mid - start); // ***

    u64 x[256];
    int nres = 0;
    u64 ncandidates = 0;

    u64 *g_de_z = malloc(p * fixed_size * sizeof(u64));
    u64 *z_buff = malloc(p * fixed_size * sizeof(u64));

    u64 *reception_z   = calloc(p * fixed_size, sizeof(u64));
    u64 *reception_g_z = calloc(p * fixed_size, sizeof(u64));

    for (int num = 0; num < nb_chunk; num++)
    {
        int *nb_demandes_p = calloc(sizeof(int), p);

        int debut_chunk = debut + (local_size / nb_chunk) * num;
        int fin_chunk   = debut + (local_size / nb_chunk) * (num + 1)
                          + (num == nb_chunk - 1 ? local_size % nb_chunk : 0);

        for (u64 z = debut_chunk; z < fin_chunk; z++)
        {
            u64 y = g(z);
            int owner = y % p;
            g_de_z [owner * fixed_size + nb_demandes_p[owner] + 1] = y;
            z_buff [owner * fixed_size + nb_demandes_p[owner] + 1] = z;
            nb_demandes_p[owner]++;
        }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < p; i++)
        {
            g_de_z [i * fixed_size] = nb_demandes_p[i];
            z_buff[i * fixed_size]  = nb_demandes_p[i];
        }

        MPI_Alltoall(g_de_z, fixed_size, MPI_UNSIGNED_LONG,
                     reception_g_z, fixed_size, MPI_UNSIGNED_LONG,
                     MPI_COMM_WORLD);
        MPI_Alltoall(z_buff, fixed_size, MPI_UNSIGNED_LONG,
                     reception_z, fixed_size, MPI_UNSIGNED_LONG,
                     MPI_COMM_WORLD);

        int condition = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ncandidates, nres)
        #endif
        for (int i = 0; i < p; i++)
        {
            int taille;
            if (i != my_rank)
            {
                taille = reception_g_z[i * fixed_size];
            }
            else
            {
                taille = nb_demandes_p[my_rank];
            }

            for (int j = 1; j < taille + 1; j++)
            {
                if (condition) continue;  
                u64 y, z;
                if (i == my_rank)
                {
                    y = g_de_z [i * fixed_size + j];
                    z = z_buff  [i * fixed_size + j];
                }
                else
                {
                    y = reception_g_z[i * fixed_size + j];
                    z = reception_z  [i * fixed_size + j];
                }

                int nx = dict_probe(y, 256, x);
                assert(nx >= 0);
                ncandidates += nx;

                for (int a = 0; a < nx; a++)
                {
                    if (is_good_pair(x[a], z))
                    {
                        if (nres == maxres) 
                        {
                            condition = 1;
                            break;
                        }
                        k1[nres] = x[a];
                        k2[nres] = z;
                        nres++;
                    }
                }
            }
        }
        if (condition) return -1;
        free(nb_demandes_p);
    }

    // *** printf("fin"); // ***
    // *** print_memory_usage(); // ***

    free(reception_g_z);
    free(reception_z);
    free(g_de_z);
    free(z_buff);

    return nres;
}

int main(int argc, char **argv)
{
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    process_command_line_options(argc, argv);

    mask = (1 << n) - 1;

    size = 1ull << n;

    local_size = size / p;
    surplus = size % p;

    if (my_rank == p - 1)
    {
        local_size += surplus;
    }

    nb_chunk = max(1, min(100, size / (p * 1000)));
    if (my_rank == 0)
        printf("nb chunk = %d\n", nb_chunk);

    fixed_size = (((size / p + surplus) / p) + 5) * 1.3 / nb_chunk;

    u64 k1[16], k2[16];

    // On mesure localement le temps avec clock()
    __clock_t start = clock();
    int nkey = golden_claw_search(16, k1, k2);
    __clock_t end = clock();
    __clock_t local_time = (end - start);

    //printf("temps pris proc %d = %ld\n", my_rank, end - start);

    // On obtient la mémoire locale maxRSS
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long local_mem = usage.ru_maxrss;

    // Réduction pour obtenir le maximum sur tous les rangs
    __clock_t global_max_time;
    MPI_Reduce(&local_time, &global_max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    long global_max_mem;
    MPI_Reduce(&local_mem, &global_max_mem, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    for (int i = 0; i < nkey; i++) {
         assert(f(k1[i]) == g(k2[i]));
         assert(is_good_pair(k1[i], k2[i]));
         printf("Solution found: (%" PRIx64 ", %" PRIx64 ") [checked OK]\n", k1[i], k2[i]);
    }

    // Affiche seulement une fois sur le rang 0
    if (my_rank == 0)
    {
        printf("\nMax execution time among all processes = %lld (clock ticks)\n",
               (long long)global_max_time);
        printf("Max memory usage among all processes    = %ld KB\n\n",
               global_max_mem);
    }

    MPI_Finalize();
    return 0;
}
