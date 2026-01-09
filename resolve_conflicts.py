import os

def resolve_file_conflicts(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    new_lines = []
    in_conflict = False
    in_remote = False
    resolved_count = 0
    
    # Validation: check if file actually has markers
    has_markers = False
    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            has_markers = True
            break
            
    if not has_markers:
        return

    print(f"Resolving conflicts in: {filepath}")

    for line in lines:
        if line.startswith('<<<<<<< HEAD'):
            in_conflict = True
            in_remote = False
            # We are now in the local (HEAD) block. We KEEP these lines.
            continue
        
        if line.startswith('======='):
            if in_conflict:
                in_remote = True
                # We are now in the remote block. We DISCARD these lines.
                continue
        
        if line.startswith('>>>>>>>'):
            if in_conflict:
                in_conflict = False
                in_remote = False
                resolved_count += 1
                # End of conflict block
                continue

        # Logic for keeping/discarding lines
        if in_conflict:
            if not in_remote:
                # Keep HEAD content
                new_lines.append(line)
            else:
                # Discard Remote content
                pass
        else:
            # Outside of conflict, keep line
            new_lines.append(line)

    # Write back the resolved content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print(f"  Fixed {resolved_count} conflict blocks.")

def scan_and_resolve(root_dir):
    for root, dirs, files in os.walk(root_dir):
        # Skip git dir
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            filepath = os.path.join(root, file)
            # Only process text files usually, but our robust reader handles binaries by skipping/ignoring errors hopefully or simplistic check
            # For this project, mostly code files
            if file.endswith(('.py', '.html', '.js', '.css', '.md', '.txt', '.json')):
                resolve_file_conflicts(filepath)

if __name__ == "__main__":
    scan_and_resolve('.')
