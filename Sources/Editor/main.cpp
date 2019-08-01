#include "SyrinxEditor.h"

int main(int argc, char *argv[])
{
    Syrinx::Editor editor;
    editor.init(800, 600);
    editor.run();
    
    return 0;
}
