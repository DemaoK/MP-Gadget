/* Tests for MP-Gadget BigFile snapshot IO. */

#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#define _XOPEN_SOURCE 700

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <ftw.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <bigfile-mpi.h>

#include "stub.h"

#include <libgadget/cosmology.h>
#include <libgadget/partmanager.h>
#include <libgadget/petaio.h>
#include <libgadget/physconst.h>
#include <libgadget/slotsmanager.h>
#include <libgadget/utils/endrun.h>
#include <libgadget/utils/mymalloc.h>
#include <libgadget/utils/paramset.h>

static int
remove_tree_entry(const char * path, const struct stat * st, int typeflag, struct FTW * ftwbuf)
{
    (void) st;
    (void) typeflag;
    (void) ftwbuf;
    return remove(path);
}

static void
remove_tree(const char * path)
{
    if(path[0])
        nftw(path, remove_tree_entry, 64, FTW_DEPTH | FTW_PHYS);
}

static void
declare_petaio_params(ParameterSet * ps)
{
    param_declare_int(ps, "BytesPerFile", OPTIONAL, 1024 * 1024, NULL);
    param_declare_int(ps, "NumWriters", OPTIONAL, 0, NULL);
    param_declare_int(ps, "MinNumWriters", OPTIONAL, 1, NULL);
    param_declare_int(ps, "WritersPerFile", OPTIONAL, 8, NULL);
    param_declare_int(ps, "EnableAggregatedIO", OPTIONAL, 0, NULL);
    param_declare_int(ps, "AggregatedIOThreshold", OPTIONAL, 256, NULL);
    param_declare_int(ps, "OutputPotential", OPTIONAL, 0, NULL);
    param_declare_int(ps, "OutputTimebins", OPTIONAL, 0, NULL);
    param_declare_int(ps, "OutputHeliumFractions", OPTIONAL, 0, NULL);
    param_declare_string(ps, "SnapshotFileBase", OPTIONAL, "PART", NULL);
    param_declare_string(ps, "InitCondFile", REQUIRED, NULL, NULL);
    param_declare_int(ps, "ExcursionSetReionOn", OPTIONAL, 0, NULL);
}

static void
set_test_petaio_params(const char * icpath)
{
    ParameterSet * ps = parameter_set_new();
    declare_petaio_params(ps);

    char content[1024];
    snprintf(content, sizeof(content), "InitCondFile %s\n", icpath);
    assert_int_equal(param_parse(ps, content), 0);
    assert_int_equal(param_validate(ps), 0);
    set_petaio_params(ps);
    petaio_init();
    parameter_set_free(ps);
}

static void
local_range(const size_t ntotal, const int rank, const int nrank, size_t * start, size_t * end)
{
    *start = ntotal * (size_t) rank / (size_t) nrank;
    *end = ntotal * (size_t) (rank + 1) / (size_t) nrank;
}

static void
write_block(BigFile * bf, const char * name, const char * dtype, const int items,
            void * data, const size_t nlocal, const size_t ntotal, MPI_Comm comm)
{
    int nrank;
    MPI_Comm_size(comm, &nrank);
    int nfile = nrank;
    if(ntotal < (size_t) nfile)
        nfile = (int) ntotal;
    if(nfile < 1)
        nfile = 1;

    BigBlock block;
    BigBlockPtr ptr;
    BigArray array;
    size_t dims[2] = {nlocal, (size_t) items};

    big_array_init(&array, data, dtype, 2, dims, NULL);

    if(0 != big_file_mpi_create_block(bf, &block, name, dtype, items, nfile, ntotal, comm))
        endrun(0, "Failed to create block %s: %s\n", name, big_file_get_error_message());
    if(0 != big_block_seek(&block, &ptr, 0))
        endrun(0, "Failed to seek block %s: %s\n", name, big_file_get_error_message());
    if(0 != big_block_mpi_write(&block, &ptr, &array, nfile, comm))
        endrun(0, "Failed to write block %s: %s\n", name, big_file_get_error_message());
    if(0 != big_block_mpi_close(&block, comm))
        endrun(0, "Failed to close block %s: %s\n", name, big_file_get_error_message());
}

static void
write_header(BigFile * bf)
{
    BigBlock header;
    int64_t npart[6] = {0, 2, 0, 3, 0, 0};
    double mass[6] = {0, 11.0, 0, 0, 1.0, 1.0};
    double time = 0.5;
    double box = 100.0;
    int use_peculiar_velocity = 1;
    int zoom_highres = DMMASK;
    int zoom_boundary = 1 << 3;
    double omega0 = 0.3;
    double omega_lambda = 0.7;
    double omega_baryon = 0.0;
    double omega_ur = 0.0;
    double omega_k = 0.0;
    double omega_fld = 0.0;
    double w0_fld = -1.0;
    double wa_fld = 0.0;
    double hubble = 0.7;
    int class_radiation = 0;
    double cmb = 2.7255;
    double unit_length = 3.085678e21;
    double unit_mass = 1.989e43;
    double unit_velocity = 1e5;

    if(0 != big_file_mpi_create_block(bf, &header, "Header", NULL, 0, 0, 0, MPI_COMM_WORLD))
        endrun(0, "Failed to create Header: %s\n", big_file_get_error_message());

    int failed =
        (0 != big_block_set_attr(&header, "TotNumPart", npart, "u8", 6)) ||
        (0 != big_block_set_attr(&header, "TotNumPartInit", npart, "u8", 6)) ||
        (0 != big_block_set_attr(&header, "MassTable", mass, "f8", 6)) ||
        (0 != big_block_set_attr(&header, "Time", &time, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "TimeIC", &time, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "BoxSize", &box, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "UsePeculiarVelocity", &use_peculiar_velocity, "i4", 1)) ||
        (0 != big_block_set_attr(&header, "Omega0", &omega0, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "OmegaLambda", &omega_lambda, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "OmegaBaryon", &omega_baryon, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "OmegaUR", &omega_ur, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "OmegaK", &omega_k, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "OmegaFld", &omega_fld, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "W0_Fld", &w0_fld, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "WA_Fld", &wa_fld, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "HubbleParam", &hubble, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "class_radiation_convention", &class_radiation, "i4", 1)) ||
        (0 != big_block_set_attr(&header, "CMBTemperature", &cmb, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "UnitLength_in_cm", &unit_length, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "UnitMass_in_g", &unit_mass, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "UnitVelocity_in_cm_per_s", &unit_velocity, "f8", 1)) ||
        (0 != big_block_set_attr(&header, "ZoomHighResTypes", &zoom_highres, "i4", 1)) ||
        (0 != big_block_set_attr(&header, "ZoomBoundaryTypes", &zoom_boundary, "i4", 1));
    if(failed)
        endrun(0, "Failed to set Header attrs: %s\n", big_file_get_error_message());

    if(0 != big_block_mpi_close(&header, MPI_COMM_WORLD))
        endrun(0, "Failed to close Header: %s\n", big_file_get_error_message());
}

static void
write_particle_type(BigFile * bf, const int ptype, const size_t ntotal, const int write_mass, MPI_Comm comm)
{
    int rank, nrank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nrank);

    size_t start, end;
    local_range(ntotal, rank, nrank, &start, &end);
    const size_t nlocal = end - start;

    double (* pos)[3] = calloc(nlocal ? nlocal : 1, sizeof(*pos));
    float (* vel)[3] = calloc(nlocal ? nlocal : 1, sizeof(*vel));
    uint64_t * ids = calloc(nlocal ? nlocal : 1, sizeof(*ids));
    float * mass = calloc(nlocal ? nlocal : 1, sizeof(*mass));

    size_t i;
    for(i = 0; i < nlocal; i++) {
        const size_t g = start + i;
        pos[i][0] = 1.0 + ptype + g;
        pos[i][1] = 2.0 + ptype + g;
        pos[i][2] = 3.0 + ptype + g;
        ids[i] = (uint64_t) (1000 + 100 * ptype + g);
        mass[i] = (float) (100.0 + 10.0 * ptype + g);
    }

    char name[64];
    snprintf(name, sizeof(name), "%d/Position", ptype);
    write_block(bf, name, "f8", 3, pos, nlocal, ntotal, comm);
    snprintf(name, sizeof(name), "%d/Velocity", ptype);
    write_block(bf, name, "f4", 3, vel, nlocal, ntotal, comm);
    snprintf(name, sizeof(name), "%d/ID", ptype);
    write_block(bf, name, "u8", 1, ids, nlocal, ntotal, comm);
    if(write_mass) {
        snprintf(name, sizeof(name), "%d/Mass", ptype);
        write_block(bf, name, "f4", 1, mass, nlocal, ntotal, comm);
    }

    free(pos);
    free(vel);
    free(ids);
    free(mass);
}

static void
write_zoom_ic(const char * path)
{
    BigFile bf = {0};
    if(0 != big_file_mpi_create(&bf, path, MPI_COMM_WORLD))
        endrun(0, "Failed to create IC at %s: %s\n", path, big_file_get_error_message());

    write_header(&bf);
    write_particle_type(&bf, 1, 2, 0, MPI_COMM_WORLD);
    write_particle_type(&bf, 3, 3, 1, MPI_COMM_WORLD);

    if(0 != big_file_mpi_close(&bf, MPI_COMM_WORLD))
        endrun(0, "Failed to close IC at %s: %s\n", path, big_file_get_error_message());
}

static void
setup_particle_memory(struct header_data * header)
{
    int rank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    int ptype;
    int64_t nlocal_total = 0;
    for(ptype = 0; ptype < 6; ptype++) {
        size_t start, end;
        local_range((size_t) header->NTotal[ptype], rank, nrank, &start, &end);
        header->NLocal[ptype] = (int64_t) (end - start);
        nlocal_total += header->NLocal[ptype];
    }

    particle_alloc_memory(PartManager, header->BoxSize, nlocal_total + 8);
    PartManager->NumPart = nlocal_total;
    slots_init(1.5, SlotsManager);
    slots_setup_topology(PartManager, header->NLocal, SlotsManager);
}

static void
assert_zoom_ic_loaded(const struct header_data * header)
{
    assert_int_equal(header->ZoomHighResTypes, DMMASK);
    assert_int_equal(header->ZoomBoundaryTypes, 1 << 3);
    assert_int_equal(header->ZoomTypeMasksPresent, 1);

    int64_t offset = 0;
    int ptype;
    int rank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    for(ptype = 0; ptype < 6; ptype++) {
        size_t start, end;
        local_range((size_t) header->NTotal[ptype], rank, nrank, &start, &end);
        int64_t i;
        for(i = 0; i < header->NLocal[ptype]; i++) {
            const struct particle_data * part = &P[offset + i];
            const size_t g = start + (size_t) i;
            assert_int_equal(part->Type, ptype);
            assert_int_equal(part->ID, (uint64_t) (1000 + 100 * ptype + g));
            assert_true(fabs(part->Pos[0] - (1.0 + ptype + g)) < 1e-12);
            if(ptype == 1)
                assert_true(fabs(part->Mass - 11.0) < 1e-6);
            if(ptype == 3)
                assert_true(fabs(part->Mass - (100.0 + 10.0 * ptype + g)) < 1e-6);
        }
        offset += header->NLocal[ptype];
    }
}

static void
test_zoom_ic_mass_blocks(void ** state)
{
    (void) state;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char tmpdir[256] = "";
    if(rank == 0) {
        snprintf(tmpdir, sizeof(tmpdir), "/tmp/mpgadget-petaio-XXXXXX");
        assert_non_null(mkdtemp(tmpdir));
    }
    MPI_Bcast(tmpdir, sizeof(tmpdir), MPI_CHAR, 0, MPI_COMM_WORLD);

    char icpath[512];
    snprintf(icpath, sizeof(icpath), "%s/ic", tmpdir);

    set_test_petaio_params(icpath);
    write_zoom_ic(icpath);

    Cosmology CP = {0};
    CP.CMBTemperature = 2.7255;
    CP.w0_fld = -1.0;
    CP.wa_fld = 0.0;
    struct header_data header = petaio_read_header(-1, ".", &CP);
    const struct UnitSystem units = get_unitsystem(header.UnitLength_in_cm, header.UnitMass_in_g,
                                                  header.UnitVelocity_in_cm_per_s);
    init_cosmology(&CP, header.TimeIC, units);

    setup_particle_memory(&header);
    petaio_read_snapshot(-1, ".", &CP, &header, PartManager, SlotsManager, MPI_COMM_WORLD);
    assert_zoom_ic_loaded(&header);

    myfree(PartManager->Base);
    memset(PartManager, 0, sizeof(PartManager[0]));
    memset(SlotsManager, 0, sizeof(SlotsManager[0]));

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
        remove_tree(tmpdir);
    MPI_Barrier(MPI_COMM_WORLD);
}

int
main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zoom_ic_mass_blocks),
    };
    return cmocka_run_group_tests_mpi(tests, NULL, NULL);
}
