#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <libdwarf.h>


struct node {
    uintptr_t start;
    uintptr_t end;
    char *name;
    int namelen;
    struct node *child;
    int child_len;
};

void spaces(int n)
{
    for( int i = 0; i < n; i++ ) {
        printf(" ");
    }
}

void print_tree (struct node n, int level)
{
    spaces(2*level); printf("node:  %s\n", n.name);
    spaces(2*level); printf("start: %"PRIxPTR"\n", n.start);
    spaces(2*level); printf("end:   %"PRIxPTR"\n", n.end);

    for( int i = 0; i < n.child_len; i++ ) {
        print_tree(n.child[i], level+1);
    }
}

int check_tree (struct node n)
{
    if (n.start > n.end) {
        return 1;
    }

    for (int i = 0; i < n.child_len; i++) {
        if (n.child[i].start < n.start ||
                n.child[i].end > n.end ||
                check_tree(n.child[i])) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char *argv)
{

    //struct node tree[10];
    //a[0].child_len = 1;
    //a[0].name = "none";
    //a[0].start = 0;
    //a[0].end = 0xffffffffffffffff;

    struct node root;
    root.child_len = 1;
    root.name = "none";
    root.start = 0;
    root.end = 0xffffffffffffffff;

    struct node main;
    main.child_len = 1;
    main.name = "main.c:main";
    main.start = 0x1234;
    main.end = 0x2345;

    struct node vadd;
    vadd.child_len = 0;
    vadd.name = "main.c:vadd";
    vadd.start = 0x1235;
    vadd.end = 0x2345;

    struct node vmul;
    vmul.child_len = 0;
    vmul.name = "main.c:vmul";
    vmul.start = 0x1235;
    vmul.end = 0x2345;

    root.child = &main;
    main.child = &vadd;
    //main.child = &vmul;

    if (check_tree(root)) {
        printf("Error in tree!\n");
        exit(1);
    } else {
        printf("Looks good!\n");
    }

    print_tree(root, 0);
}
