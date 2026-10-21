#ifndef CONSENRICH_BARCODE_SELECTION_H
#define CONSENRICH_BARCODE_SELECTION_H

#include <htslib/sam.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct crBarcodeSelection
{
    char tag[3];
    char **barcodes;
    size_t count;
    int active;
} crBarcodeSelection;

static int crBarcodeCompare(const void *left, const void *right)
{
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static void crBarcodeSelectionDestroy(crBarcodeSelection *selection)
{
    size_t i;
    if (selection == NULL)
        return;
    for (i = 0; i < selection->count; ++i)
        free(selection->barcodes[i]);
    free(selection->barcodes);
    free(selection);
}

static crBarcodeSelection *crBarcodeSelectionCreate(
    const char *tag, const char *path, const char **error)
{
    crBarcodeSelection *selection = (crBarcodeSelection *)calloc(1, sizeof(crBarcodeSelection));
    FILE *fileHandle;
    char *line = NULL;
    size_t capacity = 0;
    size_t barcodeCapacity = 0;
    ssize_t length;
    char **expanded;
    if (selection == NULL)
    {
        *error = "failed to allocate barcode selection";
        return NULL;
    }
    if (path == NULL || path[0] == '\0')
        return selection;
    selection->active = 1;
    if (tag == NULL || strlen(tag) != 2 ||
        !((tag[0] >= 'A' && tag[0] <= 'Z') || (tag[0] >= 'a' && tag[0] <= 'z')) ||
        !((tag[1] >= 'A' && tag[1] <= 'Z') || (tag[1] >= 'a' && tag[1] <= 'z') ||
          (tag[1] >= '0' && tag[1] <= '9')))
    {
        *error = "barcodeTag must be a two-character SAM tag";
        crBarcodeSelectionDestroy(selection);
        return NULL;
    }
    memcpy(selection->tag, tag, 2);
    fileHandle = fopen(path, "r");
    if (fileHandle == NULL)
    {
        *error = "failed to open barcode allowlist";
        crBarcodeSelectionDestroy(selection);
        return NULL;
    }
    while ((length = getline(&line, &capacity, fileHandle)) >= 0)
    {
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = '\0';
        if (length == 0)
            continue;
        if (selection->count == barcodeCapacity)
        {
            barcodeCapacity = barcodeCapacity == 0 ? 256 : barcodeCapacity * 2;
            expanded = (char **)realloc(selection->barcodes, barcodeCapacity * sizeof(char *));
            if (expanded == NULL)
                break;
            selection->barcodes = expanded;
        }
        selection->barcodes[selection->count] = strdup(line);
        if (selection->barcodes[selection->count] == NULL)
            break;
        ++selection->count;
    }
    free(line);
    if (!feof(fileHandle))
    {
        fclose(fileHandle);
        *error = "failed to read barcode allowlist";
        crBarcodeSelectionDestroy(selection);
        return NULL;
    }
    fclose(fileHandle);
    if (selection->count > 1)
        qsort(selection->barcodes, selection->count, sizeof(char *), crBarcodeCompare);
    return selection;
}

static int crBarcodeSelected(const crBarcodeSelection *selection, const bam1_t *record)
{
    uint8_t *value;
    char *barcode;
    if (selection == NULL || !selection->active)
        return 1;
    if (selection->count == 0)
        return 0;
    value = bam_aux_get(record, selection->tag);
    if (value == NULL)
        return 0;
    if (value[0] != 'Z')
        return -1;
    barcode = (char *)(value + 1);
    if (barcode[0] == '\0')
        return 0;
    return bsearch(&barcode, selection->barcodes, selection->count,
                   sizeof(char *), crBarcodeCompare) != NULL;
}

#endif
